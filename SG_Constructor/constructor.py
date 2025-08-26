import os
import json
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import re

# --- Configuration ---
part_dir = 'C:/PartLevelProject/part_seg_dataset_after_vlm/samples'
part_seg_data_dir = 'C:/PartLevelProject/part_seg_dataset_after_vlm/part_seg_dataset'
kaf_dir = 'C:/PartLevelProject/indoor/KAF_out/results/vg_minitest'
part_seg_base = "C:/PartLevelProject/part_seg_dataset_after_vlm/part_seg_dataset/part_seg_dataset.json"
output_filename = "scene_graph_3.json"

reference_size = (512, 512)

# --- Helper Functions ---

def resize_image_and_update_json(image_path, json_data, reference_size):
    """
    Resizes an image and updates the bounding box coordinates in the corresponding JSON data.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return json_data

    try:
        img_cv = cv2.imread(image_path)
        original_size = (img_cv.shape[1], img_cv.shape[0])
        if original_size == reference_size:
            return json_data # No need to resize or update
        
        height, width = original_size
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0
        transform_matrix = get_affine_transform(
            center, scale, 0, reference_size
        )

        # 2. APPLY the transformation to the image
        warped_img = cv2.warpAffine(
            img_cv,
            transform_matrix,
            reference_size,  # (width, height) for output size
            flags=cv2.INTER_LINEAR
        )

        # 3. SAVE the newly transformed image
        cv2.imwrite(image_path, warped_img)

        width_ratio = reference_size[0] / original_size[0]
        height_ratio = reference_size[1] / original_size[1]

        if 'masks' in json_data:
            for level in json_data.get('masks', {}):
                for item in json_data['masks'][level]:
                    if 'bbox' in item and len(item['bbox']) == 4:
                        bbox = item['bbox']
                        item['bbox'] = [
                            int(bbox[0] * width_ratio),
                            int(bbox[1] * height_ratio),
                            int(bbox[2] * width_ratio),
                            int(bbox[3] * height_ratio)
                        ]
        
        if 'bbox' in json_data and isinstance(json_data['bbox'], list):
             for i, bbox in enumerate(json_data['bbox']):
                    if len(bbox) == 4:
                        json_data['bbox'][i] = [
                            int(bbox[0] * width_ratio),
                            int(bbox[1] * height_ratio),
                            int(bbox[2] * width_ratio),
                            int(bbox[3] * height_ratio)
                        ]

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

    return json_data

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    The boxes are expected in [xmin, ymin, xmax, ymax] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def get_mask_bbox(mask_path):
    """
    Calculate bounding box from mask PNG file.
    Returns bbox in [xmin, ymin, xmax, ymax] format.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    
    rows, cols = np.where(mask > 0)
    if len(rows) == 0 or len(cols) == 0: return None
    
    min_x, max_x = np.min(cols), np.max(cols)
    min_y, max_y = np.min(rows), np.max(rows)
    return [int(min_x), int(min_y), int(max_x), int(max_y)]

def flatten_part_masks(mask_node, path=""):
    """
    Recursively flattens the hierarchical mask structure from part_seg_data.
    """
    current_path = f"{mask_node['path']}" if path else mask_node.get('path', 'unknown')
    
    masks = []
    if "bbox" in mask_node:
        node_copy = mask_node.copy()
        node_copy['path'] = current_path
        masks.append(node_copy)

    if "children" in mask_node:
        for child_key in mask_node["children"]:
            child_node = mask_node["children"][child_key]
            masks.extend(flatten_part_masks(child_node, current_path))
            
    return masks

def parse_part_id(part_id):
    """Extract sample directory and mask name from part_id for scene graph construction."""
    parts = part_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].startswith("mask"):
        return parts[0], parts[1]
    return None, None

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    def get_dir(src_point, rot_rad):
        _sin, _cos = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * _cos - src_point[1] * _sin
        src_result[1] = src_point[0] * _sin + src_point[1] * _cos

        return src_result

    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

    



# --- Pipeline Steps ---

def resize_ssam_data(part_seg_path, ref_size):
    """ Step 1a: Resize SSAM images and update corresponding JSON. """
    print("--- Step 1a: Resizing SSAM data ---")

    # --- Part 1: Process the part segmentation data ---
    def resize_part_nodes_recursive(node, base_image_dir, ref_size):
        """
        A local helper function to recursively traverse the part hierarchy,
        resizing each mask image and updating its bounding box.
        """
        relative_path = node.get("path")
        bbox = node.get("bbox")

        # Process the current node if it has a path and a valid bbox
        if relative_path and isinstance(bbox, list) and len(bbox) == 4:
            image_path = os.path.join(base_image_dir, relative_path)
            if os.path.exists(image_path):
                try:
                    img_cv = cv2.imread(image_path)
                    original_size = (img_cv.shape[1], img_cv.shape[0])
                    height, width = original_size
                    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
                    scale = max(height, width) * 1.0
                    if original_size != ref_size:
                        # Resize and save the image file
                        transform_matrix = get_affine_transform(
                            center, scale, 0, ref_size
                        )

                        # 2. APPLY the transformation to the image
                        warped_img = cv2.warpAffine(
                            img_cv,
                            transform_matrix,
                            ref_size,  # (width, height) for output size
                            flags=cv2.INTER_LINEAR
                        )

                        # 3. SAVE the newly transformed image
                        cv2.imwrite(image_path, warped_img)

                        # Calculate scaling ratios and update the bbox in the current node
                        width_ratio = ref_size[0] / original_size[0]
                        height_ratio = ref_size[1] / original_size[1]
                        
                        node['bbox'] = [
                            int(bbox[0] * width_ratio),
                            int(bbox[1] * height_ratio),
                            int(bbox[2] * width_ratio),
                            int(bbox[3] * height_ratio)
                        ]
                except Exception as e:
                    print(f"Warning: Could not process part image {image_path}: {e}")

        # Always recurse to process children nodes
        if "children" in node:
            for child_node in node["children"].values():
                resize_part_nodes_recursive(child_node, base_image_dir, ref_size)

    with open(part_seg_path, 'r+') as f:
        part_data = json.load(f)
        # Iterate through each sample ID (e.g., "id 1") in the top level of the JSON
        for sample_id, sample_data in part_data.items():
            # Define the base directory where images for this sample are stored
            base_image_dir = os.path.join(part_seg_data_dir, sample_id)

            # Start the recursion for each top-level mask
            masks_hierarchy = sample_data.get("masks", {})
            for top_level_mask_node in masks_hierarchy.values():
                resize_part_nodes_recursive(top_level_mask_node, base_image_dir, ref_size)
        
        # After processing all samples, write the modified data back to the file
        f.seek(0)
        json.dump(part_data, f, indent=4)
        f.truncate()
        
    print("--- SSAM data resizing complete ---")

def resize_sample_images(samples_dir, ref_size):
    """ Step 1b: Resizes all PNG images in the samples directory. """
    print(f"--- Step 1b: Resizing sample images in {samples_dir} ---")
    if not os.path.exists(samples_dir):
        print(f"Warning: Samples directory not found at {samples_dir}"); return

    for sample_subdir in os.listdir(samples_dir):
        subdir_path = os.path.join(samples_dir, sample_subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith('.png'):
                    image_path = os.path.join(subdir_path, filename)
                    try:
                        with Image.open(image_path) as img:
                            if img.size != ref_size:
                                original_size = img.size
                                height, width = original_size
                                center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
                                scale = max(height, width) * 1.0
                                trans_img = get_affine_transform(
                                    center, scale, 0, [512, 512]
                                )
                                trans_img.save(image_path)
                    except Exception as e:
                        print(f"Error resizing image {image_path}: {e}")
    print("--- Sample image resizing complete ---")

def resize_kaf_data(kaf_root_dir, ref_size):
    """ Step 2: Resize KAF images and update corresponding JSON. """
    print("--- Step 2: Resizing KAF data ---")
    for scene_dir in os.listdir(kaf_root_dir):
        scene_path = os.path.join(kaf_root_dir, scene_dir)
        if os.path.isdir(scene_path):
            pred_graph_path = os.path.join(scene_path, 'pred_graph.json')
            if os.path.exists(pred_graph_path):
                with open(pred_graph_path, 'r+') as f:
                    kaf_data = json.load(f)
                    image_path = os.path.join(scene_path, 'original.jpg') 
                    resize_image_and_update_json(image_path, kaf_data, ref_size)
                    f.seek(0); json.dump(kaf_data, f, indent=4); f.truncate()
    print("--- KAF data resizing complete ---")
    
def get_top_two_levels_mask_items(masks_node):
    """
    Collects mask items from the top two levels of the hierarchy (top-level
    masks and their direct children). This prevents matching deeper, more
    granular parts.
    """
    items = []
    if not isinstance(masks_node, dict):
        return items

    # Level 1: Iterate through top-level masks (e.g., "mask1", "mask9")
    for l1_node in masks_node.values():
        if "bbox" in l1_node:
            items.append(l1_node)
        
        # Level 2: Iterate through the children of the top-level masks
        if "children" in l1_node and isinstance(l1_node["children"], dict):
            for l2_node in l1_node["children"].values():
                if "bbox" in l2_node:
                    items.append(l2_node)
    return items

def match_ssam_to_kaf(part_seg_path, kaf_root_dir):
    """
    Step 3: Match SSAM objects with KAF objects and save inferred data.
    
    This version matches only the top two levels of the SSAM mask hierarchy
    to ensure only significant parts are compared.
    """
    print("--- Step 3: Matching SSAM to KAF and saving inferred files ---")
    
    try:
        with open(part_seg_path, "r") as f:
            part_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {part_seg_path}: {e}")
        return

    # Iterate through the dictionary of images
    for image_id, image_entry in part_data.items():
        # Construct the path to the corresponding KAF directory
        id_part = f'50-{image_id}'
        kaf_target_dir = os.path.join(kaf_root_dir, id_part)
        
        kaf_graph_path = os.path.join(kaf_target_dir, 'pred_graph.json')
        if not os.path.exists(kaf_graph_path):
            # If the KAF file doesn't exist, we skip this entry
            continue

        with open(kaf_graph_path, "r") as f:
            kaf_data = json.load(f)

        # Use the helper to get a flat list of items from the top two levels
        all_level_items = get_top_two_levels_mask_items(image_entry.get("masks", {}))
        
        kaf_objects = kaf_data.get('objects', [])
        kaf_bboxes = kaf_data.get('bbox', [])
        
        used_kaf_indices = set()
        item_matches = []
        
        for item in all_level_items:
            item_bbox = item.get("bbox")
            if not item_bbox:
                continue
            
            best_kaf_match = None
            max_iou = -1.0
            best_kaf_idx = -1
            
            for kaf_idx, kaf_obj in enumerate(kaf_objects):
                if kaf_idx >= len(kaf_bboxes):
                    continue
                kaf_bbox = kaf_bboxes[kaf_idx]
                iou = calculate_iou(item_bbox, kaf_bbox)
                if iou > max_iou:
                    max_iou = iou
                    best_kaf_idx = kaf_idx
                    idx = f"id {item.get('path', '-1')}; {kaf_idx}"
                    best_kaf_match = {"name": kaf_obj.get("name", "unknown"), "index": idx, "iou": iou}
            
            # Use a threshold to consider a match valid
            if best_kaf_match and max_iou > 0.5:
                item_matches.append((item, best_kaf_match, best_kaf_idx, max_iou))
        
        # Sort by IoU to prioritize assigning the best matches first
        item_matches.sort(key=lambda x: x[3], reverse=True)
        
        # Assign unique matches
        for item, best_kaf_match, best_kaf_idx, _ in item_matches:
            if best_kaf_idx not in used_kaf_indices:
                # The 'item' is a reference to the dictionary within part_data,
                # so modifying it updates the main data structure.
                item["matched_kaf_name"] = best_kaf_match["name"]
                item["matched_kaf_index"] = best_kaf_match["index"]
                item["overlap_score"] = best_kaf_match["iou"]
                used_kaf_indices.add(best_kaf_idx)

    # After processing all images, write the updated dictionary back to the file
    print(f"--- Writing all updated scene data back to {part_seg_path} ---")
    with open(part_seg_path, "w") as f:
        # Dump the entire modified dictionary, preserving its structure
        json.dump(part_data, f, indent=4)

    print("--- SSAM to KAF matching complete ---")

    
def match_part_to_samples(part_seg_path, samples_dir):
    """ Step 4: Match part segmentation with samples (with scene-aware uniqueness). """
    ## TODO: add the father part seg to match the samples
    print("--- Step 6: Matching Part Seg to Samples ---")
    with open(part_seg_path, 'r') as f: part_data = json.load(f)
    outputDict = {}
    outputBiDict = {}
    # Create a global list of all parts with scene-aware unique identifiers
    all_parts = []
    for top_key in part_data:
        if "masks" in part_data[top_key]:
            for mask_key in part_data[top_key]["masks"]:
                parts = flatten_part_masks(part_data[top_key]["masks"][mask_key])
                for idx, part in enumerate(parts):
                    part['_source_scene'] = top_key
                    part['_unique_id'] = f"{top_key}_{part.get('path', f'unknown_{idx}')}"
                all_parts.extend(parts)

    # Track used parts globally with scene-aware IDs
    global_used_part_ids = set()
    
    # Collect all potential matches
    all_matches = []
    
    for sample_dir in os.listdir(samples_dir):
        sample_path = os.path.join(samples_dir, sample_dir)
        if not os.path.isdir(sample_path): continue

        for mask_file in os.listdir(sample_path):
            if not (mask_file.lower().endswith(".png") and mask_file.lower().startswith("mask")): continue
            
            mask_path = os.path.join(sample_path, mask_file)
            mask_bbox = get_mask_bbox(mask_path)
            if mask_bbox is None: continue

            best_match = None
            max_iou = -1.0
            best_part_unique_id = None
            
            for part in all_parts:
                unique_id = part['_unique_id']
                if unique_id in global_used_part_ids:  # Skip already used parts
                    continue
                    
                part_bbox_xywh = part.get("bbox")
                if not part_bbox_xywh: continue
                part_bbox = [part_bbox_xywh[0], part_bbox_xywh[1], part_bbox_xywh[2], part_bbox_xywh[3]]
                iou = calculate_iou(mask_bbox, part_bbox)
                
                if iou > max_iou:
                    max_iou = iou
                    best_match = part
                    best_part_unique_id = unique_id
            
            # if best_match and max_iou > 0.5:
            if best_match:
                all_matches.append((mask_path, best_match, best_part_unique_id, max_iou))
    
    # Sort by IoU and assign uniquely
    all_matches.sort(key=lambda x: x[3], reverse=True)
    
    successful_matches = 0
    duplicate_attempts = 0
    
    for mask_path, best_match, best_part_unique_id, max_iou in all_matches:
        if best_part_unique_id not in global_used_part_ids:
            part_id = best_match.get('path', 'unknown')
            outputDict[best_part_unique_id] = mask_path
            outputBiDict[mask_path] = best_part_unique_id
            global_used_part_ids.add(best_part_unique_id)
            successful_matches += 1
            
            sample_dir = os.path.basename(os.path.dirname(mask_path))
            mask_file = os.path.basename(mask_path)
            source_scene = best_match.get('_source_scene', 'unknown')
            print(f"Sample {mask_file} from {sample_dir} matched with part {part_id} from scene {source_scene} with IoU: {max_iou}")
        else:
            duplicate_attempts += 1
    
    print(f"--- Part-Sample matching: {successful_matches} successful, {duplicate_attempts} duplicates prevented ---")
    print("--- Part Seg to Samples matching complete ---")
    return outputDict, outputBiDict



def create_scene_graph(kaf_root_dir, samples_dir, part_seg_path, part_to_sample_mapping, sample_to_part_mapping, out_filename):
    """ Step 5: Create the final detailed scene graph using the pre-computed mapping. """
    print("--- Step 5: Creating Final Scene Graph ---")
    try:
        with open(part_seg_path, 'r') as f: part_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse {part_seg_path}: {e}"); return

    def build_part_hierarchy(template_node, scene_id, part_path_to_sample_path, sample_to_part_mapping, samples_dir):
        parts_list = []
        uniqueIDList = []
        relationList = []
        if not isinstance(template_node, dict): return parts_list
        matchedRelation = []
        for _, part_info in template_node.items():
            part_full_path = part_info.get("path")
            if not part_full_path: continue
        
            unique_id_key = f"{scene_id}_{part_full_path}"
            uniqueIDList.append(unique_id_key)
            matched_sample_mask_path = part_path_to_sample_path.get(unique_id_key)
            
            part_desc, sample_dir_name = "N/A (unmatched)", "unknown"
            
            if matched_sample_mask_path:
                sample_dir_name = os.path.basename(os.path.dirname(matched_sample_mask_path))
                sample_mask_name = os.path.splitext(os.path.basename(matched_sample_mask_path))[0]
                config_path = os.path.join(samples_dir, sample_dir_name, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f: config_data = json.load(f)
                    part_desc = "N/A"
                    for rel in config_data.get("kinematic relation", []):
                        if len(rel) < 3: continue
                        details, rel_part0, rel_part1 = rel[2], rel[0], rel[1]
                        if rel_part0 == sample_mask_name: part_desc = details.get("part0_desc", part_desc)
                        if rel_part1 == sample_mask_name: part_desc = details.get("part1_desc", part_desc)
                        outputDetail = details
                        rootPath = os.path.dirname(matched_sample_mask_path)
                        unique_id_key0 = os.path.join(rootPath, rel_part0+'.png')
                        unique_id_key1 = os.path.join(rootPath, rel_part1+'.png')
                        relationDict = {
                            "subject": sample_to_part_mapping.get(unique_id_key0),
                            "object": sample_to_part_mapping.get(unique_id_key1),
                            }
                        if relationDict not in matchedRelation:
                            matchedRelation.append(relationDict)
                            outputDetail.update(relationDict)
                            relationList.append(outputDetail)
            parts, relations = build_part_hierarchy(part_info.get("children", {}), scene_id, part_path_to_sample_path, sample_to_part_mapping, samples_dir)
            new_part = {
                "id": part_full_path,
                "description": part_desc,
                "children": parts,
                "kinematic_relations": relations
            }
            parts_list.append(new_part)
        return parts_list, relationList

    for scene_id, scene_data in part_data.items():
        kaf_scene_dir_name = f'50-{scene_id}'
        scene_path = os.path.join(kaf_root_dir, kaf_scene_dir_name)
        if not os.path.isdir(scene_path):
            print(f"Warning: KAF directory not found for scene {scene_id}, skipping."); continue
        
        final_objects = []
        top_level_instances = [inst for inst in scene_data.get("masks", {}).values() if "matched_kaf_name" in inst]

        for inst in top_level_instances:
            parts, relationList = build_part_hierarchy(inst.get("children", {}), scene_id, part_to_sample_mapping, sample_to_part_mapping, samples_dir)
            inst_description = inst.get("description", "")
            obj = {
                "id": inst.get("path", "unknown"),
                "kaf_name": inst.get("matched_kaf_name", "unknown"),
                "instance description": inst_description, 
                "kaf_index": inst.get("matched_kaf_index", -1),
                "bbox": inst.get("bbox"),
                "parts": parts,
                "kinematic_relations": relationList,
            }
            final_objects.append(obj)

        relationships = []
        pred_graph_path = os.path.join(scene_path, 'pred_graph.json')
        if os.path.exists(pred_graph_path):
            # final_obj_map = {o["kaf_index"]: i for i, o in enumerate(final_objects) if o["kaf_index"] != -1}
            final_obj_map = {}
            for item in final_objects:
                kafIdx = item["kaf_index"]
                if kafIdx != -1:
                    idx = kafIdx.split('; ')[-1]
                    final_obj_map.update({idx:item["id"]})
            with open(pred_graph_path, 'r') as f: pred_graph = json.load(f)
            for rel in pred_graph.get("relationships", []):
                subj_idx, obj_idx = rel.get("subject"), rel.get("object")
                subj_key = final_obj_map.get(str(subj_idx))
                obj_key = final_obj_map.get(str(obj_idx))
                if subj_key is not None and obj_key is not None:
                    relationships.append({"predicate": rel.get("predicate"), "subject": subj_key, "object": obj_key})

        scene_graph = {"image_path": os.path.join(scene_path, 'original.jpg'), "objects": final_objects, "relationships": relationships}
        with open(os.path.join(scene_path, out_filename), 'w') as f: json.dump(scene_graph, f, indent=4)
        print(f"Created scene graph for {scene_id}")
        
    print("--- Scene Graph construction complete ---")

def main():
    """ Main function to run the entire pipeline. """
    # Step 1: Resize all image data and update corresponding JSON files
    resize_ssam_data(part_seg_base, ref_size=reference_size)
    resize_sample_images(part_dir, reference_size)
    resize_kaf_data(kaf_dir, reference_size)

    # Step 2: Match SSAM objects with KAF objects
    match_ssam_to_kaf(part_seg_base, kaf_dir)

    # Step 3: Match parts to samples and get the mapping dictionary.
    outputDict, outputBiDict = match_part_to_samples(part_seg_base, part_dir)

    # Step 4: Create the final scene graphs using the mapping dictionary.
    create_scene_graph(kaf_dir, part_dir, part_seg_base, outputDict, outputBiDict, output_filename)

    print("\nPipeline finished!")

if __name__ == '__main__':
    main()
