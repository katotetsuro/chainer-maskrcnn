import chainercv


class InstanceSegmentationVOCEvaluator(chainercv.extensions.evaluator.instance_segmentation_voc_evaluator.InstanceSegmentationVOCEvaluator):
    def evaluate(self):
        print('use preset: evaluation')
        self._targets['main'].use_preset('evaluate')
        observation = super().evaluate()
        print('use preset: visualize')
        self._targets['main'].use_preset('visualize')
        return observation
