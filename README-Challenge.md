# This fork is created for implementation of the assignment from ABEJA to explore YOLO-v5 

## Step 1: Understand the requirements
- Define custom metric along with previous metrics
- Monitor custom_acc during training
- Save weights of epoch with the best custom_acc value


## Step 2: Code Exploration

Here's the tree structure for overall code:  
```
`-- yolov5
    |-- CONTRIBUTING.md
    |-- Dockerfile
    |-- LICENSE
    |-- README.md
    |-- data
    |   |-- Argoverse_HD.yaml
    |   |-- GlobalWheat2020.yaml
    |   |-- Objects365.yaml
    |   |-- SKU-110K.yaml
    |   |-- VOC.yaml
    |   |-- VisDrone.yaml
    |   |-- coco.yaml
    |   |-- coco128.yaml
    |   |-- hyps
    |   |   |-- hyp.finetune.yaml
    |   |   |-- hyp.finetune_objects365.yaml
    |   |   |-- hyp.scratch-p6.yaml
    |   |   `-- hyp.scratch.yaml
    |   |-- images
    |   |   |-- bus.jpg
    |   |   `-- zidane.jpg
    |   |-- scripts
    |   |   |-- download_weights.sh
    |   |   |-- get_coco.sh
    |   |   `-- get_coco128.sh
    |   `-- xView.yaml
    |-- detect.py
    |-- export.py
    |-- hubconf.py
    |-- models
    |   |-- __init__.py
    |   |-- common.py
    |   |-- experimental.py
    |   |-- hub
    |   |   |-- anchors.yaml
    |   |   |-- yolov3-spp.yaml
    |   |   |-- yolov3-tiny.yaml
    |   |   |-- yolov3.yaml
    |   |   |-- yolov5-fpn.yaml
    |   |   |-- yolov5-p2.yaml
    |   |   |-- yolov5-p6.yaml
    |   |   |-- yolov5-p7.yaml
    |   |   |-- yolov5-panet.yaml
    |   |   |-- yolov5l6.yaml
    |   |   |-- yolov5m6.yaml
    |   |   |-- yolov5s-transformer.yaml
    |   |   |-- yolov5s6.yaml
    |   |   `-- yolov5x6.yaml
    |   |-- yolo.py
    |   |-- yolov5l.yaml
    |   |-- yolov5m.yaml
    |   |-- yolov5s.yaml
    |   `-- yolov5x.yaml
    |-- requirements.txt
    |-- train.py
    |-- tutorial.ipynb
    |-- utils
    |   |-- __init__.py
    |   |-- activations.py
    |   |-- augmentations.py
    |   |-- autoanchor.py
    |   |-- aws
    |   |   |-- __init__.py
    |   |   |-- mime.sh
    |   |   |-- resume.py
    |   |   `-- userdata.sh
    |   |-- datasets.py
    |   |-- flask_rest_api
    |   |   |-- README.md
    |   |   |-- example_request.py
    |   |   `-- restapi.py
    |   |-- general.py
    |   |-- google_app_engine
    |   |   |-- Dockerfile
    |   |   |-- additional_requirements.txt
    |   |   `-- app.yaml
    |   |-- google_utils.py
    |   |-- loss.py
    |   |-- metrics.py
    |   |-- plots.py
    |   |-- torch_utils.py
    |   `-- wandb_logging
    |       |-- __init__.py
    |       |-- log_dataset.py
    |       |-- sweep.py
    |       |-- sweep.yaml
    |       `-- wandb_utils.py
    `-- val.py

12 directories, 77 files

```

- Explored the directory structure and files
- Among other files and directories, I found `metrics.py` file in `utils` directory, which looks quite relevant to the task at hand.
- Additionally I went through the `CONTRIBUTING.md` since I have to submit a PR, hence it would be nice to check out how do we contribute officialy to this repo. 
- Mainly `metrics.py`, `train.py`, and `val.py` needs to be updated to implement the given requirements. 


## Step 3: Analyze relevant code and update code
- After thorough analysis, I thought of 3 ways on how we can calculate the custom accuracy metric as follows:

1. Implement a class CustomAccuracy from scratch to calculate accuracy
```
class CustomAccuracy(Metric):
    """
    Reference: https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.45,
                 output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        super(CustomAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        self.nc = 0
        super(CustomAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, y_pred, y_true):
        """
        Arguments:
            y_pred (Array[N, 6]), x1, y1, x2, y2, conf, class
            y_true (Array[M, 5]), class, x1, y1, x2, y2
        """
        print('detections', y_pred, y_pred.shape)
        print('labels', y_true, y_true.shape)
        # Filtering only those detections which are more than conf
        y_pred = y_pred[y_pred[:, 4] > self.conf]
        # Getting classes from predictions and 
        y_pred = y_pred[:, 5].int()
        y_true = y_true[:, 0].int()
        print('y_true', y_true, y_true.shape)
        print('y_pred', y_pred, y_pred.shape)
        
        #########################
        Code to add for matching of the ground truth objects with predicted ones
        #########################
        
        #y_pred, y = output[0].detach(), output[1].detach()
        indices = torch.argmax(y_pred, dim=1)
        correct = torch.eq(indices, y_true).view(-1)
        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("matrix")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._num_correct.item() / self._num_examples
```


2. Calculate accuracy as part of `val.py` script where stats are appended. However, that implementation is not good from oriented-object design perspective since the whole code would be merged into `val.py` and we have to make a generic function that could be easily changed based on the given requirements.


```
if nl:
    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
    scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
    correct = process_batch(predn, labelsn, iouv)
    # Update accuracy
    # correct -> True Positive, TN -> True Negative
    custom_acc = torch.tensor([(len(correct) + tn) / nl]) 

    if plots:
        confusion_matrix.process_batch(predn, labelsn)

```

3. Write a derived function from ConfusionMatrix class based on the accuracy calculation like below
**Accuracy = (TP + TN) / (P + N)**
Where we can use diagonal for (TP + TN) for all classes and sum of matrix for (P + N)
I have used ignite metrics as base. If required we can easily update the implementation of below class or add more feature in the below class. 
```
class CustomAccuracy(Metric):
    """
    Reference: https://pytorch.org/ignite/metrics.html#how-to-create-a-custom-metric
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.45,
                 output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = None
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf, 
                                                iou_thres=self.iou_thres)
        super(CustomAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        self.nc = 0
        super(CustomAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, y_pred, y_true):
        """
        Creating wrapper of ConfusionMatrix since that class already contains
        values that we need to calculate accuracy
        Arguments:
            y_pred (Array[N, 6]), x1, y1, x2, y2, conf, class
            y_true (Array[M, 5]), class, x1, y1, x2, y2
        """
        self.confusion_matrix.process_batch(y_pred, y_true)
        self.matrix = self.confusion_matrix.matrix
        
    @sync_all_reduce("matrix")
    def compute(self):
        """
        Function to calculate accuracy based on the below formula
        Accuracy = (TP + TN) / (P + N)
        (TP + TN) could be represented as diagonal elements
        (P + N) could be represented as all elements
        # Reference: https://en.wikipedia.org/wiki/Confusion_matrix
        Returns: 
            Accuracy based on the confusion matrix values
        """
        diagonal_sum = np.sum(np.diagonal(self.matrix)) 
        total_sum = np.sum(self.matrix)
        return (diagonal_sum / total_sum) * 100

```


## Step 4: Implementation
- I tried using my own pytorch `Dockerfile` to run the code, but there were some dependency issues of nvidia-driver
- Hence I tried to use `Dockerfile` provided in repo. However, out of the box the Dockerfile doesn't work with Nvidia 2080 GPUs that I am using. Hence I made some custom changes to make it work. Please refer to the `Dockerfile` for custom changes.
- Additionally there was shared memory error while running `train.py` which I solved using `--shm-size 100G` while using `docker run` command. Reference: https://github.com/pytorch/pytorch/issues/2244#issuecomment-318864552
- Additionally, there were some issue with current pytorch version, hence I have installed specific libraries of pytorch that works with my GPU server. I removed this two library in my `requirements.txt` installation `torch>=1.7.0, torchvision>=0.8.1` to install specific versions. 
- Additional install: pip install pytorch-ignite


**Here's the current output out-of-the box for the train.py script without changing the implementation**  

Command: **python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache --device 0,1**
```
train: weights=yolov5s.pt, cfg=, data=coco128.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=3, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache_images=True, image_weights=False, device=0,1, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, entity=None, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias=latest, local_rank=-1
github: skipping check (Docker image), for updates see https://github.com/ultralytics/yolov5
YOLOv5 🚀 v5.0-309-g264be1a torch 1.7.1 CUDA:0 (GeForce RTX 2080 Ti, 11019.4375MB)
                                        CUDA:1 (GeForce RTX 2080 Ti, 11019.4375MB)

hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Downloading https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt to yolov5s.pt...
100%|█████████████████████████████████████████████████████████████████████████████████| 14.1M/14.1M [00:00<00:00, 101MB/s]


                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 283 layers, 7276605 parameters, 7276605 gradients, 17.1 GFLOPs

Transferred 362/362 items from yolov5s.pt

WARNING: Dataset not found, nonexistent paths: ['/usr/src/datasets/coco128/images/train2017']
Downloading https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip ...
100%|████████████████████████████████████████████████████████████████████████████████| 6.66M/6.66M [00:00<00:00, 27.8MB/s]
Dataset autodownload success

Scaled weight_decay = 0.0005
Optimizer groups: 62 .bias, 62 conv.weight, 59 other
DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.
See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.
train: Scanning '../datasets/coco128/labels/train2017' images and labels...128 found, 0 missing, 2 empty, 0 corrupted: 100
train: New cache created: ../datasets/coco128/labels/train2017.cache
train: Caching images (0.1GB): 100%|██████████████████████████████████████████████████| 128/128 [00:00<00:00, 1525.79it/s]
val: Scanning '../datasets/coco128/labels/train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corrupted
val: Caching images (0.1GB): 100%|████████████████████████████████████████████████████| 128/128 [00:00<00:00, 1193.83it/s]
Plotting labels... 

autoanchor: Analyzing anchors... anchors/target = 4.26, Best Possible Recall (BPR) = 0.9935
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/exp
Starting training for 3 epochs...

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       0/2     2.03G   0.04615   0.07257   0.02249    0.1412       185       640: 100%|█████| 8/8 [00:08<00:00,  1.05s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0%|       | 0/4 [00:00<?, ?it/s]/usr/src/app/val.py:60: UserWarning: This overload of nonzero is deprecated:
        nonzero()
Consider using one of the following signatures instead:
        nonzero(*, bool as_tuple) (Triggered internally at  /opt/conda/conda-bld/pytorch_1607370117127/work/torch/csrc/utils/python_arg_parser.cpp:882.)
  ti = (cls == tcls).nonzero().view(-1)  # label indices
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 4/4 [00:01<00:00,  3.81it/
                 all        128        929      0.719      0.555      0.646      0.426

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       1/2     3.03G   0.04552   0.06705   0.02234    0.1349       220       640: 100%|█████| 8/8 [00:01<00:00,  5.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 4/4 [00:01<00:00,  3.81it/
                 all        128        929      0.706      0.558      0.651      0.428

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       2/2     3.03G   0.04435   0.06943   0.02142    0.1352       201       640: 100%|█████| 8/8 [00:01<00:00,  5.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|█| 4/4 [00:02<00:00,  1.75it/
                 all        128        929      0.721      0.555      0.656      0.436
3 epochs completed in 0.006 hours.

Optimizer stripped from runs/train/exp/weights/last.pt, 14.8MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.8MB
```




## Step 5: Validate if everything looks good
After several iteration of code update, I validated if all the implementation was fine. There were some issues in implementation especially to figure out the variable names and their meaning. But finally I was able to make my CustomAccuracy class work. 

Below is the final output of the implemented CustomAccuracy metric class.  

Command: **python train.py --img 640 --batch 16 --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device 0,1**
```
train: weights=yolov5s.pt, cfg=, data=coco128.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=10, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache_images=True, image_weights=False, device=0,1, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, entity=None, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, upload_dataset=False, bbox_interval=-1, save_period=-1, artifact_alias=latest, local_rank=-1
github: skipping check (Docker image), for updates see https://github.com/ultralytics/yolov5
YOLOv5 🚀 v5.0-309-g264be1a torch 1.7.1 CUDA:0 (GeForce RTX 2080 Ti, 11019.4375MB)
                                        CUDA:1 (GeForce RTX 2080 Ti, 11019.4375MB)

hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 283 layers, 7276605 parameters, 7276605 gradients, 17.1 GFLOPs

Transferred 362/362 items from yolov5s.pt
Scaled weight_decay = 0.0005
Optimizer groups: 62 .bias, 62 conv.weight, 59 other
DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.
See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.
train: Scanning '../datasets/coco128/labels/train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100%|█| 12
train: Caching images (0.1GB): 100%|████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 1082.03it/s]
val: Scanning '../datasets/coco128/labels/train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100%|█| 128/
val: Caching images (0.1GB): 100%|███████████████████████████████████████████████████████████████████| 128/128 [00:00<00:00, 991.21it/s]
Plotting labels... 

autoanchor: Analyzing anchors... anchors/target = 4.25, Best Possible Recall (BPR) = 0.9935
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/exp24
Starting training for 10 epochs...

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       0/9     2.03G   0.04366    0.0693    0.0226    0.1356       163       640: 100%|███████████████████| 8/8 [00:06<00:00,  1.24it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc:   0%|          | 0/4 [00:00<?, ?it/s]/data/storage07/user_data/daultanidine01/Documents/pytorch-practice/yolo-exploration/yolov5/val.py:60: UserWarning: This overload of nonzero is deprecated:
        nonzero()
Consider using one of the following signatures instead:
        nonzero(*, bool as_tuple) (Triggered internally at  /opt/conda/conda-bld/pytorch_1607370117127/work/torch/csrc/utils/python_arg_parser.cpp:882.)
  ti = (cls == tcls).nonzero().view(-1)  # label indices
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.10it/s]
                 all        128        929      0.662      0.581      0.644      0.424       46.6
Saving best accuracy model with accuracy 46.6 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       1/9     3.03G   0.04543   0.06385   0.02396    0.1332       227       640: 100%|███████████████████| 8/8 [00:01<00:00,  4.74it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.14it/s]
                 all        128        929      0.728      0.553      0.654      0.429       46.5

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       2/9     3.03G   0.04436   0.07141   0.02166    0.1374       222       640: 100%|███████████████████| 8/8 [00:01<00:00,  5.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.37it/s]
                 all        128        929       0.72      0.561      0.661      0.436       46.8
Saving best accuracy model with accuracy 46.78 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       3/9     3.03G   0.04566   0.06252   0.02366    0.1318       208       640: 100%|███████████████████| 8/8 [00:01<00:00,  5.01it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  2.95it/s]
                 all        128        929      0.724      0.559      0.666      0.442       47.2
Saving best accuracy model with accuracy 47.23 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       4/9     3.03G    0.0453   0.06513   0.02154     0.132       265       640: 100%|███████████████████| 8/8 [00:01<00:00,  4.88it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.39it/s]
                 all        128        929      0.687      0.594      0.667      0.444       47.9
Saving best accuracy model with accuracy 47.94 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       5/9     3.03G   0.04442   0.06766   0.02098    0.1331       221       640: 100%|███████████████████| 8/8 [00:01<00:00,  5.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.38it/s]
                 all        128        929      0.644      0.636      0.672      0.447       48.4
Saving best accuracy model with accuracy 48.39 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       6/9     3.03G     0.044   0.05701   0.02189    0.1229       181       640: 100%|███████████████████| 8/8 [00:01<00:00,  4.97it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.45it/s]
                 all        128        929      0.645      0.635      0.673      0.447       48.1

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       7/9     3.03G   0.04638    0.0659   0.01998    0.1323       127       640: 100%|███████████████████| 8/8 [00:01<00:00,  5.08it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.14it/s]
                 all        128        929       0.72      0.595      0.681      0.452       48.7
Saving best accuracy model with accuracy 48.68 %!

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       8/9     3.03G   0.04209   0.06809   0.02063    0.1308       174       640: 100%|███████████████████| 8/8 [00:01<00:00,  4.73it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:01<00:00,  3.35it/s]
                 all        128        929      0.722      0.599      0.682      0.453       48.6

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
       9/9     3.03G   0.04402   0.06502   0.02079    0.1298       169       640: 100%|███████████████████| 8/8 [00:01<00:00,  5.13it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95        acc: 100%|██| 4/4 [00:02<00:00,  1.49it/s]
                 all        128        929      0.645      0.657      0.684      0.456       48.7
Saving best accuracy model with accuracy 48.69 %!
10 epochs completed in 0.017 hours.

Optimizer stripped from runs/train/exp24/weights/last.pt, 14.8MB
Optimizer stripped from runs/train/exp24/weights/best.pt, 14.8MB
```
Few things to note here is as follows:
- **Model weights are saved whenever accuracy is greater than the previous best accuracy.**
- **Also, we can monitor accuracy while model training in shell as well as `tensorboard` with new metric `metrics/custom_acc`**
- Model accuracy is increasing over each epoch along with mAP metric. This helps us to make sure that everything is working well.
- There are two ways where we can use CustomAccuracy as follows:
    - **Optimize based on just accuracy and save model**
    - **Optimize in fitness function with given weight** - This will help us to evolve our model based on accuracy along with previous metrics. Currently the weight is `0.1` for accuracy in `fitness function`. `w = [0.0, 0.0, 0.1, 0.8, 0.1]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95, accuracy]`. However, we could change it based on the tuning of the weights. 


## Step 6: Push the PR to forked repo

