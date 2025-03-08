DetectionResult(
    detections=[
        Detection(
            bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
            categories=[
                Category(index=None, score=0.780297040939331, display_name=None, category_name='cat')
            ], keypoints=[]), 
        Detection(bounding_box=BoundingBox(origin_x=303, origin_y=27, width=249, height=345), categories=[Category(index=None, score=0.7625645399093628, display_name=None, category_name='dog')], keypoints=[])
    ]
)
print(detection_result.detections[0].categories[0].category_name)
print(detection_result.detections[1].categories[0].category_name)