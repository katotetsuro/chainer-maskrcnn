import chainercv


class InstanceSegmentationVOCEvaluator(chainercv.extensions.evaluator.instance_segmentation_voc_evaluator.InstanceSegmentationVOCEvaluator):
    def evaluate(self):
        self._targets['main'].faster_rcnn.use_preset('evaluate')
        observation = super().evaluate()
        self._targets['main'].faster_rcnn.use_preset('visualize')
        return observation
