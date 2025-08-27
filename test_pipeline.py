import os
import torch

from KAF_Generation.tools.custom_predict import *
from pixtral_12B_Inference.relation_generator import VLMRelationGenerator
from SG_Constructor.constructor import *
from SayPlan_Reconstruct.pipeline import Pipeline

# INPUT: SSAM
part_seg_data_dir = '' # SSAM output root directory
part_seg_base = "" # SSAM output root json
# KAF_Generation Configs
fcsgg_folder_name = "C:/PartLevelProject/indoor/part_seg_dataset" # KAF output root directory
fcsgg_out_dir = "C:/PartLevelProject/indoor/KAF_out"
fcsgg_base_name = "results"
fcsgg_dataset_name = "vg_minitest" 
# VLM Configs
vlm_dataset_dir = part_seg_base
vlm_src_image_dir = part_seg_data_dir
vlm_output_dir = 'C:/PartLevelProject/indoor/vlm_out'
vlm_processed_dataset_dir = ''
# SG CONSTRUCTOR Configs
reference_size = (512, 512)
sg_constructor_part_dir = vlm_output_dir
sg_constructor_part_seg_data_dir = part_seg_data_dir
sg_constructor_fcsgg_dir = os.path.join(fcsgg_out_dir, fcsgg_base_name, fcsgg_dataset_name)
sg_constructor_part_seg_base = part_seg_base
sg_constructor_output_filename = "scene_graph.json"
# SAYPLAN Configs
sgPath = '' ## FIXME: Enter sg path of a given scene id here
task = '' ## FIXME: Enter task here
sayplan_jsonPath = '' ## FIXME: Enter sg with part kinematic relations

def run_fcsgg():
    
    cfg = setup()

    ###########    build model   ###########

    predictor = DefaultPredictor(cfg)
    out_dir = os.path.join(fcsgg_out_dir, fcsgg_base_name, fcsgg_dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset_metadata = MetadataCatalog.get(fcsgg_dataset_name)
    image_files = [os.path.join(fcsgg_folder_name, f) 
                  for f in os.listdir(fcsgg_folder_name) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file_name in image_files:
        with torch.no_grad():
            d = {"file_name": file_name}
            output = predictor(cv2.imread(file_name))
            visualize_detections(d, output, fcsgg_out_dir, dataset_metadata)
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
    run_fcsgg()
    
    run_vlm_inference()
    
    resize_ssam_data(sg_constructor_part_seg_base, ref_size=reference_size)
    resize_sample_images(sg_constructor_part_dir, reference_size)
    resize_kaf_data(sg_constructor_fcsgg_dir, reference_size)
    match_ssam_to_kaf(sg_constructor_part_seg_base, sg_constructor_fcsgg_dir)
    outputDict, outputBiDict = match_part_to_samples(sg_constructor_part_seg_base, sg_constructor_part_dir)
    create_scene_graph(sg_constructor_fcsgg_dir, sg_constructor_part_dir, sg_constructor_part_seg_base, outputDict, outputBiDict, sg_constructor_output_filename)

    pipeline = Pipeline(sgPath, task)
    
    pipeline.prune_graph()
    
    keptInstanceList = pipeline.keptSG
    # Access the exact node objects starting with this kept instance list
    
    plan = pipeline.plan()
    
    ## FIXME: KAF_Net, output the sg with part kinematic relationships to sayplan_jsonPath
    
    pipeline.AddKinematicRelations(sayplan_jsonPath)
    replan = pipeline.replan(plan)
