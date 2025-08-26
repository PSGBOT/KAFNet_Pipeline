import os
import torch

from KAF_Generation.tools.custom_predict import *
from pixtral_12B_Inference.relation_generator import VLMRelationGenerator
from SG_Constructor.constructor import *
from SayPlan_Reconstruct.pipeline import Pipeline

# INPUT: SSAM
part_seg_data_dir = '' # SSAM output root directory
part_seg_base = "" # SSAM output root json
# KAF Configs
kaf_folder_name = "C:/PartLevelProject/indoor/part_seg_dataset" # KAF output root directory
kaf_out_dir = "C:/PartLevelProject/indoor/KAF_out"
kaf_base_name = "results"
kaf_dataset_name = "vg_minitest" 
# VLM Configs
vlm_dataset_dir = ''
vlm_src_image_dir =''
vlm_output_dir = ''
vlm_processed_dataset_dir = ''
# SG CONSTRUCTOR Configs
sg_constructor_part_dir = vlm_processed_dataset_dir
sg_constructor_part_seg_data_dir = part_seg_data_dir
sg_constructor_kaf_dir = os.path.join(kaf_out_dir, kaf_base_name, kaf_dataset_name)
sg_constructor_part_seg_base = part_seg_base
sg_constructor_output_filename = "scene_graph_3.json"
# SAYPLAN Configs
sgPath = ''
task = ''
sgKinematicPath=''

def run_kaf():
    
    cfg = setup()

    ###########    build model   ###########

    predictor = DefaultPredictor(cfg)
    out_dir = os.path.join(kaf_out_dir, kaf_base_name, kaf_dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset_metadata = MetadataCatalog.get(kaf_dataset_name)
    image_files = [os.path.join(kaf_folder_name, f) 
                  for f in os.listdir(kaf_folder_name) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file_name in image_files:
        with torch.no_grad():
            d = {"file_name": file_name}
            output = predictor(cv2.imread(file_name))
            visualize_detections(d, output, kaf_out_dir, dataset_metadata)
            # break

def run_vlm_inference():
    generator = VLMRelationGenerator(
        vlm_dataset_dir,
        vlm_src_image_dir,
        vlm_output_dir,
        vlm_processed_dataset_dir,
    )

    # Load dataset
    generator.load_dataset()

    generator.generate_relation(debug=False)

if __name__ == "__main__":
    run_kaf()
    pipeline = Pipeline(sgPath, task)
    run_vlm_inference()
