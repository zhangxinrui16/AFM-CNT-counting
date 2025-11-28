from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
from streamlit.web import cli as stcli
from skimage import exposure, filters, measure, morphology, util
from skimage.draw import line as draw_line

Array2D = np.ndarray


def load_grayscale(image: Image.Image) -> Array2D:
    gray = image.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    return arr


def crop_afm_region(image: Array2D) -> Tuple[Array2D, Tuple[slice, slice]]:
    """Automatically crop away outer text/legend margins.

    The AFM scan typically occupies a central rectangle, while scale bars or
    annotations live near the borders. We detect the high-variance band of rows
    and columns and keep only their bounding box.
    """

    norm = image / 255.0 if image.max() > 1 else image

    # Use high-frequency energy + gradient to highlight textured scan regions
    # while keeping legend text (narrow, low-area) suppressed.
    highpass = norm - filters.gaussian(norm, sigma=2.5)
    energy = np.abs(highpass) + filters.sobel(norm)
    energy = exposure.rescale_intensity(energy, out_range=(0.0, 1.0))

    thresh = max(np.percentile(energy, 75), energy.mean() + energy.std() * 0.5)
    mask = energy > thresh
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = morphology.remove_small_objects(mask, min_size=int(image.size * 0.005))
    mask = morphology.remove_small_holes(mask, area_threshold=int(image.size * 0.01))

    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    if regions:
        main = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = main.bbox
        pad = 4
        top = max(0, minr - pad)
        bottom = min(image.shape[0], maxr + pad)
        left = max(0, minc - pad)
        right = min(image.shape[1], maxc + pad)
    else:
        # Fallback: retain original heuristic bounds if no component is found
        blurred = filters.gaussian(norm, sigma=1.0)
        row_activity = np.std(blurred, axis=1)
        col_activity = np.std(blurred, axis=0)

        def active_bounds(profile: Array2D) -> Tuple[int, int]:
            thresh_local = max(np.percentile(profile, 60) * 0.6, profile.max() * 0.08)
            active = np.where(profile > thresh_local)[0]
            if len(active) == 0:
                return 0, len(profile)
            return int(active[0]), int(active[-1] + 1)

        top, bottom = active_bounds(row_activity)
        left, right = active_bounds(col_activity)

    cropped = image[top:bottom, left:right]
    return cropped, (slice(top, bottom), slice(left, right))


def preprocess(image: Array2D) -> Array2D:
    rescaled = exposure.rescale_intensity(
        image,
        in_range=(np.percentile(image, 1), np.percentile(image, 99)),
        out_range=(0.0, 1.0),
    )
    blurred = filters.gaussian(rescaled, sigma=1.0)
    return np.clip(blurred, 0.0, 1.0)


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
                ny, nx_coord = y + dy, x + dx
                if (ny, nx_coord) in g:
                    g.add_edge((y, x), (ny, nx_coord))
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


def visualize_detection(
    image: Array2D, skeleton: Array2D, groups: List[List[int]], segments: List[List[Tuple[int, int]]]
):
    base = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
    colors = plt.cm.get_cmap("tab20", len(groups) + 1)

    # Plotly's Image trace expects a colormodel from an explicit channel array; expand grayscale to RGB
    base_rgb = np.stack([base] * 3, axis=-1)
    base_uint8 = (base_rgb * 255).astype(np.uint8)

    fig = go.Figure()
    # 使用 px.imshow 风格的底图以确保像素坐标与散点一致
    fig.add_trace(
        go.Image(
            z=base_uint8,
            colormodel="rgb",
            hoverinfo="skip",
            name="AFM",
            opacity=0.9,
        )
    )

    for gid, seg_indices in enumerate(groups):
        xs: List[float] = []
        ys: List[float] = []
        for idx in seg_indices:
            seg = segments[idx]
            if len(seg) == 0:
                continue
            ys.extend([p[0] for p in seg])
            xs.extend([p[1] for p in seg])
            ys.append(np.nan)
            xs.append(np.nan)

        if not xs:
            continue

        rgb_vals = [int(v) for v in np.round(np.array(colors(gid)[:3]) * 255)]
        color_str = f"rgb({rgb_vals[0]},{rgb_vals[1]},{rgb_vals[2]})"
        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines+markers",
                line=dict(color=color_str, width=4),
                marker=dict(color=color_str, size=6, line=dict(width=1, color="white")),
                hovertemplate="<b>碳纳米管 %{customdata}</b><extra></extra>",
                name=f"Tube {gid + 1}",
                customdata=np.full(len(xs), gid + 1),
                hoverlabel=dict(bgcolor=color_str, font=dict(color="white")),
                opacity=0.95,
            )
        )

    fig.update_layout(
        title="检测结果（悬停高亮单根碳纳米管）",
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="closest",
        xaxis=dict(showgrid=False, visible=False, constrain="domain"),
        yaxis=dict(showgrid=False, visible=False, scaleanchor="x", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(hoverinfo="text")
    return fig


def process_image(uploaded: Image.Image):
    grayscale = load_grayscale(uploaded)
    cropped, _ = crop_afm_region(grayscale)
    pre = preprocess(cropped)
    ridges = ridge_enhance(pre)
    mask = threshold_ridges(ridges)
    bridged = connect_gaps(mask)
    skeleton = morphology.skeletonize(bridged)
    groups, segments = count_tubes(skeleton)
    return cropped, skeleton, groups, segments


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
    cropped_gray, skeleton, groups, segments = process_image(image)
    display_crop = np.clip(
        cropped_gray / 255.0 if cropped_gray.max() > 1 else cropped_gray, 0.0, 1.0
    )

    area = length_um * width_um
    tube_count = len(groups)
    density = tube_count / area if area > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("裁剪后的AFM区域")
        st.image(display_crop, caption="已去除边缘文字", use_container_width=True)
        st.markdown(
            f"**面积：** {area:.3f} μm²  |  **根数：** {tube_count}  |  **密度：** {density:.2f} 根/μm²"
        )
    with col2:
        fig = visualize_detection(cropped_gray, skeleton, groups, segments)
        st.subheader("检测结果")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

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
