from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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

    The AFM scan typically sits inside a bright rectangular frame; we seek the
    darkest rectangle bounded by white borders. A profile-based locator finds
    the longest band of non-white rows/cols and trims bright borders, while a
    high-frequency rescue falls back to the prior heuristic when needed.
    """

    norm = image / 255.0 if image.max() > 1 else image
    smooth = filters.gaussian(norm, sigma=1.0)

    def longest_run(mask: Array2D) -> Tuple[int, int]:
        start = None
        best = (0, 0)
        for idx, val in enumerate(mask):
            if val and start is None:
                start = idx
            if (not val or idx == len(mask) - 1) and start is not None:
                end_idx = idx + 1 if val and idx == len(mask) - 1 else idx
                if end_idx - start > best[1] - best[0]:
                    best = (start, end_idx)
                start = None
        if best[1] == 0:
            return 0, len(mask)
        return best

    # Locate the dominant dark band in rows/cols (non-white content).
    row_activity = (smooth < 0.97).mean(axis=1)
    col_activity = (smooth < 0.97).mean(axis=0)
    row_run = longest_run(row_activity > 0.01)
    col_run = longest_run(col_activity > 0.01)

    top, bottom = row_run
    left, right = col_run

    # Trim away any residual white borders hugging the edges.
    bright_thresh = 0.985
    row_mean = smooth.mean(axis=1)
    col_mean = smooth.mean(axis=0)
    while top < bottom - 1 and row_mean[top] > bright_thresh:
        top += 1
    while bottom - 1 > top and row_mean[bottom - 1] > bright_thresh:
        bottom -= 1
    while left < right - 1 and col_mean[left] > bright_thresh:
        left += 1
    while right - 1 > left and col_mean[right - 1] > bright_thresh:
        right -= 1

    # Guardrail: ensure the run is sufficiently large; otherwise fall back.
    min_height = int(image.shape[0] * 0.25)
    min_width = int(image.shape[1] * 0.25)

    def heuristic_crop() -> Tuple[int, int, int, int]:
        content_mask = smooth < 0.94
        content_mask = morphology.binary_closing(content_mask, morphology.disk(2))
        content_mask = morphology.remove_small_objects(
            content_mask, min_size=int(image.size * 0.002)
        )
        content_mask = morphology.remove_small_holes(
            content_mask, area_threshold=int(image.size * 0.004)
        )

        highpass = norm - filters.gaussian(norm, sigma=2.0)
        energy = np.abs(highpass) + filters.sobel(norm)
        energy = exposure.rescale_intensity(energy, out_range=(0.0, 1.0))
        energy_mask = energy > (energy.mean() + energy.std() * 0.25)

        combined = morphology.binary_dilation(content_mask | energy_mask, morphology.disk(2))
        combined = morphology.remove_small_objects(
            combined, min_size=int(image.size * 0.003)
        )

        labels = measure.label(combined)
        regions = measure.regionprops(labels)
        if not regions:
            return 0, image.shape[0], 0, image.shape[1]

        main = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = main.bbox
        pad = int(min(image.shape) * 0.04)
        top_h = max(0, minr - pad)
        bottom_h = min(image.shape[0], maxr + pad)
        left_h = max(0, minc - pad)
        right_h = min(image.shape[1], maxc + pad)
        return top_h, bottom_h, left_h, right_h

    if (bottom - top) < min_height or (right - left) < min_width:
        top, bottom, left, right = heuristic_crop()

    # Final sanity to avoid degenerate crops.
    if bottom - top < 5 or right - left < 5:
        top, bottom, left, right = 0, image.shape[0], 0, image.shape[1]

    cropped = image[top:bottom, left:right]
    return cropped, (slice(top, bottom), slice(left, right))


def preprocess(image: Array2D) -> Array2D:
    rescaled = exposure.rescale_intensity(
        image,
        in_range=(np.percentile(image, 1), np.percentile(image, 99)),
        out_range=(0.0, 1.0),
    )
    enhanced = exposure.equalize_adapthist(rescaled, clip_limit=0.02)
    blurred = filters.gaussian(enhanced, sigma=0.8)
    return np.clip(blurred, 0.0, 1.0)


def ridge_enhance(image: Array2D) -> Array2D:
    frangi = filters.frangi(image, scale_range=(1, 3), scale_step=1, beta=0.5, gamma=15)
    return exposure.rescale_intensity(frangi, out_range=(0.0, 1.0))


def threshold_ridges(ridge_map: Array2D) -> Array2D:
    thresh = filters.threshold_otsu(ridge_map)
    relaxed = max(thresh * 0.82, ridge_map.mean() + ridge_map.std() * 0.08)
    mask = ridge_map > relaxed
    clean = morphology.binary_closing(mask, morphology.disk(1))
    clean = morphology.remove_small_objects(clean, min_size=6)
    clean = morphology.remove_small_holes(clean, area_threshold=12)
    return clean


def connect_gaps(binary: Array2D, max_dist: int = 35, angle_tol: float = 25) -> Array2D:
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
    overlay = np.stack([base] * 3, axis=-1)
    colors = plt.cm.get_cmap("tab20", len(groups) + 1)

    for gid, seg_indices in enumerate(groups):
        tube_mask = np.zeros_like(image, dtype=bool)
        for idx in seg_indices:
            seg = segments[idx]
            for y, x in seg:
                tube_mask[y, x] = True
        if not tube_mask.any():
            continue

        tube_mask = morphology.dilation(tube_mask, morphology.disk(1))
        color = np.array(colors(gid)[:3])
        overlay[tube_mask] = overlay[tube_mask] * 0.35 + color * 0.65

    return np.clip(overlay, 0.0, 1.0)


def process_image(uploaded: Image.Image):
    grayscale = load_grayscale(uploaded)
    cropped, _ = crop_afm_region(grayscale)
    pre = preprocess(cropped)
    ridges = ridge_enhance(pre)
    mask = threshold_ridges(ridges)
    mask = morphology.binary_dilation(mask, morphology.disk(1))
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
        overlay = visualize_detection(cropped_gray, skeleton, groups, segments)
        st.subheader("检测结果")
        st.image(overlay, caption="彩色标注的碳纳米管", use_container_width=True)

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
