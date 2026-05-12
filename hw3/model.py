import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator # 引入 Anchor 工具

def get_model_instance_segmentation(num_classes):
    
    # 1. 設計專屬微小物體的 Anchor (每個 Tuple 對應 FPN 的一個特徵層)
    # 將預設的 32~512 縮小為 16~256
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # 2. 載入模型，加入自訂 Anchor 並解除預測數量上限
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT", 
        trainable_backbone_layers=5,
        box_detections_per_img=500,           # 解除數量封印
        rpn_anchor_generator=anchor_generator # 換上小尺寸 Anchor
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model