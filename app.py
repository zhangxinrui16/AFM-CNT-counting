import io
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.web import cli as stcli
from skimage import exposure, filters, morphology, util
from skimage.draw import line as draw_line

Array2D = np.ndarray


def load_grayscale(image: Image.Image) -> Array2D:
    gray = image.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    return arr


def preprocess(image: Array2D) -> Array2D:
    rescaled = exposure.rescale_intensity(
        image,
        in_range=(np.percentile(image, 1), np.percentile(image, 99)),
        out_range=(0.0, 1.0),
    )
    blurred = filters.gaussian(rescaled, sigma=1.0)
    return blurred


def ridge_enhance(image: Array2D) -> Array2D:
    frangi = filters.frangi(image, scale_range=(1, 3), scale_step=1, beta=0.5, gamma=15)
    return exposure.rescale_intensity(frangi, out_range=(0.0, 1.0))


def threshold_ridges(ridge_map: Array2D) -> Array2D:
    thresh = filters.threshold_otsu(ridge_map)
    mask = ridge_map > thresh
    clean = morphology.remove_small_objects(mask, min_size=30)
    clean = morphology.remove_small_holes(clean, area_threshold=30)
    return clean


def connect_gaps(binary: Array2D, max_dist: int = 8, angle_tol: float = 20) -> Array2D:
    skeleton = morphology.skeletonize(binary)
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) == 0:
        return skeleton

    def neighbors(y: int, x: int) -> List[Tuple[int, int]]:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx]:
                    nbrs.append((ny, nx))
        return nbrs

    endpoints = []
    for y, x in coords:
        deg = len(neighbors(y, x))
        if deg == 1:
            endpoints.append((y, x))

    skeleton_copy = skeleton.copy()

    def orientation(pt: Tuple[int, int]) -> np.ndarray:
        y, x = pt
        nbrs = neighbors(y, x)
        if not nbrs:
            return np.array([0.0, 0.0])
        vec = np.array([nbrs[0][0] - y, nbrs[0][1] - x], dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    for i, p1 in enumerate(endpoints):
        for p2 in endpoints[i + 1 :]:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist <= max_dist:
                o1 = orientation(p1)
                o2 = orientation(p2)
                if np.linalg.norm(o1) == 0 or np.linalg.norm(o2) == 0:
                    continue
                angle = np.degrees(np.arccos(np.clip(np.dot(o1, -o2), -1.0, 1.0)))
                if angle <= angle_tol:
                    rr, cc = draw_line(p1[0], p1[1], p2[0], p2[1])
                    skeleton_copy[rr, cc] = True
    return skeleton_copy


def build_graph(skeleton: Array2D) -> nx.Graph:
    g = nx.Graph()
    coords = np.column_stack(np.nonzero(skeleton))
    for y, x in coords:
        g.add_node((y, x))
    for y, x in coords:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in g:
                    g.add_edge((y, x), (ny, nx))
    return g


def extract_segments(graph: nx.Graph) -> List[List[Tuple[int, int]]]:
    segments: List[List[Tuple[int, int]]] = []
    visited_edges = set()

    def node_degree(node):
        return graph.degree(node)

    for node in graph.nodes:
        if node_degree(node) != 2:
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted((node, neighbor)))
                if edge in visited_edges:
                    continue
                path = [node, neighbor]
                visited_edges.add(edge)
                current = neighbor
                prev = node
                while node_degree(current) == 2:
                    nbrs = list(graph.neighbors(current))
                    nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                    edge = tuple(sorted((current, nxt)))
                    if edge in visited_edges:
                        break
                    path.append(nxt)
                    visited_edges.add(edge)
                    prev, current = current, nxt
                segments.append(path)
    return segments


def segment_direction(segment: List[Tuple[int, int]], anchor_index: int = 0) -> np.ndarray:
    if len(segment) < 2:
        return np.array([0.0, 0.0])
    anchor = segment[anchor_index]
    target = segment[1] if anchor_index == 0 else segment[-2]
    vec = np.array([target[0] - anchor[0], target[1] - anchor[1]], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def merge_segments_by_orientation(segments: List[List[Tuple[int, int]]], angle_tol: float = 35) -> List[List[int]]:
    parent = list(range(len(segments)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    endpoint_map: Dict[Tuple[int, int], List[int]] = {}
    for idx, seg in enumerate(segments):
        for endpoint in (seg[0], seg[-1]):
            endpoint_map.setdefault(endpoint, []).append(idx)

    for junction, attached in endpoint_map.items():
        if len(attached) < 2:
            continue
        vectors = {}
        for seg_idx in attached:
            seg = segments[seg_idx]
            anchor_index = 0 if seg[0] == junction else -1
            vectors[seg_idx] = segment_direction(seg, anchor_index=anchor_index)
        keys = list(attached)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                vi, vj = vectors[keys[i]], vectors[keys[j]]
                if np.linalg.norm(vi) == 0 or np.linalg.norm(vj) == 0:
                    continue
                angle = np.degrees(np.arccos(np.clip(np.dot(vi, vj), -1.0, 1.0)))
                if angle < angle_tol or abs(angle - 180) < angle_tol:
                    union(keys[i], keys[j])

    groups: Dict[int, List[int]] = {}
    for idx in range(len(segments)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)
    return list(groups.values())


def count_tubes(skeleton: Array2D):
    graph = build_graph(skeleton)
    segments = extract_segments(graph)
    merged_groups = merge_segments_by_orientation(segments)
    return merged_groups, segments


def visualize_detection(image: Array2D, skeleton: Array2D, groups: List[List[int]], segments: List[List[Tuple[int, int]]]):
    base = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
    colored = np.dstack([base, base, base])
    colors = plt.cm.get_cmap("tab20", len(groups) + 1)
    for gid, seg_indices in enumerate(groups):
        color = np.array(colors(gid)[:3])
        for idx in seg_indices:
            for y, x in segments[idx]:
                if 0 <= y < colored.shape[0] and 0 <= x < colored.shape[1]:
                    colored[y, x] = color
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colored)
    ax.axis("off")
    ax.set_title("Detected carbon nanotubes")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


def process_image(uploaded: Image.Image):
    grayscale = load_grayscale(uploaded)
    pre = preprocess(grayscale)
    ridges = ridge_enhance(pre)
    mask = threshold_ridges(ridges)
    bridged = connect_gaps(mask)
    skeleton = morphology.skeletonize(bridged)
    groups, segments = count_tubes(skeleton)
    return grayscale, skeleton, groups, segments


def running_in_streamlit() -> bool:
    try:
        from streamlit import runtime
    except ImportError:
        return False

    try:
        return runtime.exists()
    except Exception:
        return False


def main():
    st.set_page_config(page_title="AFM CNT Density", layout="wide")
    st.title("AFM Carbon Nanotube Density Estimator")
    st.write(
        "上传AFM高度图像，输入扫描尺寸（微米），程序将提取碳纳米管骨架，估算根数并给出密度。"
    )

    with st.sidebar:
        st.header("输入参数")
        length_um = st.number_input("扫描长度 (μm)", min_value=0.01, value=1.0, step=0.1)
        width_um = st.number_input("扫描宽度 (μm，可选)", min_value=0.01, value=length_um, step=0.1)

    uploaded_file = st.file_uploader("上传AFM图像", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is None:
        st.info("请上传AFM高度通道图像。")
        return

    image = Image.open(uploaded_file)
    grayscale, skeleton, groups, segments = process_image(image)

    area = length_um * width_um
    tube_count = len(groups)
    density = tube_count / area if area > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始灰度图")
        st.image(grayscale, caption="输入图像灰度", use_column_width=True)
        st.markdown(
            f"**面积：** {area:.3f} μm²  |  **根数：** {tube_count}  |  **密度：** {density:.2f} 根/μm²"
        )
    with col2:
        vis_buf = visualize_detection(grayscale, skeleton, groups, segments)
        st.subheader("检测结果")
        st.image(vis_buf, caption="碳纳米管提取与计数", use_column_width=True)

    with st.expander("算法要点"):
        st.markdown(
            "- 使用Frangi滤波增强一维线状结构，抑制点状杂质。\n"
            "- Otsu阈值与连通域清理去除亮点杂质。\n"
            "- 骨架化后根据端点方向连接断裂，以拼合同一根的分段。\n"
            "- 在交叉点按方向分组：方向近似共线的段合并为一根，避免Y型交叉双计；方向差异大的段保留以正确处理X型交叉。"
        )

    st.caption("提示：如结果偏多或偏少，可调整扫描尺寸输入或在代码中调节阈值、形态学参数。")


if __name__ == "__main__":
    # Running "python app.py" lacks Streamlit's ScriptRunContext and causes warnings.
    # Delegate to "streamlit run" when needed to bootstrap the runtime cleanly.
    import sys

    if running_in_streamlit():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
