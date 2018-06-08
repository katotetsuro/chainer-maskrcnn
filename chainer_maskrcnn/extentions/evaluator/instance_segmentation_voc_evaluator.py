import chainercv


class InstanceSegmentationVOCEvaluator(chainercv.extensions.evaluator.instance_segmentation_voc_evaluator.InstanceSegmentationVOCEvaluator):
    def evaluate(self):
        self._targets['main'].use_preset('evaluate')
        observation = super().evaluate()
        self._targets['main'].use_preset('visualize')
        return observation
