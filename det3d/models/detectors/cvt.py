from ..registry import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module
class CrossViewTransformer(SingleStageDetector):
    def __init__(
        self,
        backbone,
        bbox_head,
        neck, 
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        
        super(CrossViewTransformer, self).__init__(
            backbone=backbone, bbox_head=bbox_head, neck=neck, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained
        )
    
    def forward(self, example, return_loss=True, **kwargs):
        map_view_features = self.backbone(example)
        if self.with_neck:
            map_view_features = self.neck(map_view_features)
        preds, _ = self.bbox_head(map_view_features)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)


    
