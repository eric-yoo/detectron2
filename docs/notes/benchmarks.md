
# 벤치마크

여기서는 Detectron2에서 제공하는 Mask R-CNN의 학습 속도를
다른 유명한 오픈소스 구현체 몇 가지와 비교합니다.


### 실험 환경

* 하드웨어: NVIDIA V100 8개와 NVLink.
* 소프트웨어: Python 3.7, CUDA 10.1, cuDNN 7.6.5, PyTorch 1.5,
  TensorFlow 1.15.0rc2, Keras 2.2.5, MxNet 1.6.0b20190820.
* 모델: [Detectron baseline config](https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml) 와
  동일한 하이퍼파라미터를 사용하는 엔드-투-엔드 R-50-FPN Mask-RCNN 모델  
  (scale augmentation 없음).
* 메트릭: GPU 웜업 시간을 무시하기 위해 100-500 iteration의 평균 스루풋(throughput)을 사용합니다.
  R-CNN 스타일 모델의 경우, 모델의 스루풋이 예측값(prediction)의 영향을 받기 때문에
  일반적으로 학습이 진행됨에 따라 스루풋이 변합니다. 따라서 이 메트릭을 모델 zoo의 "train speed" 
  값(전체 학습의 평균 속도)과 직접 비교하기는 어렵습니다.


### 실험 결과

```eval_rst
+-------------------------------+--------------------+
| Implementation                | Throughput (img/s) |
+===============================+====================+
| |D2| |PT|                     | 62                 |
+-------------------------------+--------------------+
| mmdetection_  |PT|            | 53                 |
+-------------------------------+--------------------+
| maskrcnn-benchmark_  |PT|     | 53                 |
+-------------------------------+--------------------+
| tensorpack_ |TF|              | 50                 |
+-------------------------------+--------------------+
| simpledet_ |mxnet|            | 39                 |
+-------------------------------+--------------------+
| Detectron_  |C2|              | 19                 |
+-------------------------------+--------------------+
| `matterport/Mask_RCNN`__ |TF| | 14                 |
+-------------------------------+--------------------+

.. _maskrcnn-benchmark: https://github.com/facebookresearch/maskrcnn-benchmark/
.. _tensorpack: https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN
.. _mmdetection: https://github.com/open-mmlab/mmdetection/
.. _simpledet: https://github.com/TuSimple/simpledet/
.. _Detectron: https://github.com/facebookresearch/Detectron
__ https://github.com/matterport/Mask_RCNN/

.. |D2| image:: https://github.com/facebookresearch/detectron2/raw/main/.github/Detectron2-Logo-Horz.svg?sanitize=true
   :height: 15pt
   :target: https://github.com/facebookresearch/detectron2/
.. |PT| image:: https://pytorch.org/assets/images/logo-icon.svg
   :width: 15pt
   :height: 15pt
   :target: https://pytorch.org
.. |TF| image:: https://static.nvidiagrid.net/ngc/containers/tensorflow.png
   :width: 15pt
   :height: 15pt
   :target: https://tensorflow.org
.. |mxnet| image:: https://github.com/dmlc/web-data/raw/master/mxnet/image/mxnet_favicon.png
   :width: 15pt
   :height: 15pt
   :target: https://mxnet.apache.org/
.. |C2| image:: https://caffe2.ai/static/logo.svg
   :width: 15pt
   :height: 15pt
   :target: https://caffe2.ai
```


각 구현체에 대한 세부 정보:

* __Detectron2__: 릴리즈 v0.1.2 에서 다음을 실행하십시오.
  ```
  python tools/train_net.py  --config-file configs/Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml --num-gpus 8
  ```

* __mmdetection__: `b0d845f` 커밋에서 다음을 실행하십시오.
  ```
  ./tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py 8
  ```

* __maskrcnn-benchmark__: `0ce8f6f` 커밋에서 PyTorch 1.5와의 호환성을 위해 `sed -i 's/torch.uint8/torch.bool/g' **/*.py; sed -i 's/AT_CHECK/TORCH_CHECK/g' **/*.cu` 를
  실행합니다. 이후 아래와 같이 학습을 실행하십시오.
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
  ```
  소프트웨어 버전 차이로 인해 maskrcnn-benchmark의 모델 zoo보다 속도가 빠른 것으로 측정됐습니다.

* __tensorpack__: `caafda` 커밋에서 `export TF_CUDNN_USE_AUTOTUNE=0` 실행 후, 다음을 실행하십시오.
  ```
  mpirun -np 8 ./train.py --config DATA.BASEDIR=/data/coco TRAINER=horovod BACKBONE.STRIDE_1X1=True TRAIN.STEPS_PER_EPOCH=50 --load ImageNet-R50-AlignPadding.npz
  ```

* __SimpleDet__: `9187a1` 커밋에서 다음을 실행하십시오.
  ```
  python detection_train.py --config config/mask_r50v1_fpn_1x.py
  ```

* __Detectron__: 다음을 실행하십시오.
  ```
  python tools/train_net.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml
  ```
  연산의 많은 부분이 CPU에서 일어나므로 성능이 제한적입니다.

* __matterport/Mask_RCNN__: `3deaec` 커밋에서 다음의 diff를 적용하고 `export TF_CUDNN_USE_AUTOTUNE=0` 실행 후, 다음을 실행하십시오.
  ```
  python coco.py train --dataset=/data/coco/ --model=imagenet
  ```
  이 구현체의 여러 세부적인 사항들은 Detectron의
  표준과 다를 수 있습니다.

  <details>
  <summary>
  (diff to make it use the same hyperparameters - click to expand)
  </summary>

  ```diff
  diff --git i/mrcnn/model.py w/mrcnn/model.py
  index 62cb2b0..61d7779 100644
  --- i/mrcnn/model.py
  +++ w/mrcnn/model.py
  @@ -2367,8 +2367,8 @@ class MaskRCNN():
        epochs=epochs,
        steps_per_epoch=self.config.STEPS_PER_EPOCH,
        callbacks=callbacks,
  -            validation_data=val_generator,
  -            validation_steps=self.config.VALIDATION_STEPS,
  +            #validation_data=val_generator,
  +            #validation_steps=self.config.VALIDATION_STEPS,
        max_queue_size=100,
        workers=workers,
        use_multiprocessing=True,
  diff --git i/mrcnn/parallel_model.py w/mrcnn/parallel_model.py
  index d2bf53b..060172a 100644
  --- i/mrcnn/parallel_model.py
  +++ w/mrcnn/parallel_model.py
  @@ -32,6 +32,7 @@ class ParallelModel(KM.Model):
      keras_model: The Keras model to parallelize
      gpu_count: Number of GPUs. Must be > 1
      """
  +        super().__init__()
      self.inner_model = keras_model
      self.gpu_count = gpu_count
      merged_outputs = self.make_parallel()
  diff --git i/samples/coco/coco.py w/samples/coco/coco.py
  index 5d172b5..239ed75 100644
  --- i/samples/coco/coco.py
  +++ w/samples/coco/coco.py
  @@ -81,7 +81,10 @@ class CocoConfig(Config):
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
  -    # GPU_COUNT = 8
  +    GPU_COUNT = 8
  +    BACKBONE = "resnet50"
  +    STEPS_PER_EPOCH = 50
  +    TRAIN_ROIS_PER_IMAGE = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
  @@ -496,29 +499,10 @@ if __name__ == '__main__':
      # *** This training schedule is an example. Update to your needs ***

      # Training - Stage 1
  -        print("Training network heads")
      model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
  -                    layers='heads',
  -                    augmentation=augmentation)
  -
  -        # Training - Stage 2
  -        # Finetune layers from ResNet stage 4 and up
  -        print("Fine tune Resnet stage 4 and up")
  -        model.train(dataset_train, dataset_val,
  -                    learning_rate=config.LEARNING_RATE,
  -                    epochs=120,
  -                    layers='4+',
  -                    augmentation=augmentation)
  -
  -        # Training - Stage 3
  -        # Fine tune all layers
  -        print("Fine tune all layers")
  -        model.train(dataset_train, dataset_val,
  -                    learning_rate=config.LEARNING_RATE / 10,
  -                    epochs=160,
  -                    layers='all',
  +                    layers='3+',
            augmentation=augmentation)

    elif args.command == "evaluate":
  ```

  </details>
