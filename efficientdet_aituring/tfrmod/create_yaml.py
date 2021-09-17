import yaml


def create_yaml(list_classes, path_output, use_data_augmentation):
    dict_classes = {(id_ + 1): name_class_ for id_,
                                               name_class_ in enumerate(list_classes)}
    dict_file = [{'num_classes': len(list_classes) + 1,
                  # 'var_freeze_expr': str('(efficientnet |fpn_cells|resample_p6)'),
                  'label_map': dict_classes,
                  'learning_rate': 0.004,
                  'lr_warmup_init': 0.0004,
                  # 'use_augmix': use_data_augmentation,
                  'nms_configs': {
                      'method': 'gaussian',
                      'iou_thresh': None,
                      'score_thresh': None,
                      'sigma': None,
                      'max_nms_inputs': 0,
                      'max_output_size': 300
                  }
                  }]
    with open(path_output, 'w') as file:
        _ = yaml.dump(dict_file, file)
