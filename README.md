# Weight-Initialization
This repository is an implementation of an efficent weight initialization method used to improve the satellite image classification. Comparative analyses with existing weight initialization techniques made on various CNN models reveal that the proposed weight initialization method outperforms the previous competitive techniques in terms of classification accuracy. 

# Abstract
Significantly increased interest in satellite images has triggered the need for efficient mechanisms for extracting useful information from massive satellite images to provide better insight into them. Even though deep learning has shown significant progress in image classification. Nevertheless, in the literature, only a few results can be found on weight initialization techniques. These techniques train the networks' weights on massive datasets and fine-tune the weights of pre-trained networks. In this study, a novel weight initialization method is proposed in the context of satellite image classification. The proposed weight initialization method is mathematically detailed during the forward and backward passes of the CNN model. Extensive experiments are carried out using six real-world datasets. Comparative analyses with existing weight initialization techniques made on various pre-trained CNN models reveal that the proposed weight initialization technique outperforms the previous competitive techniques in classification accuracy.

# Results
The proposed weight initialization method is applied to three pre-trained models, namely Resnet152V2, VGG19, and MobileNetV2. The models were trained for 100 epochs, each consisting of 32 batches. Xavier, He, and the proposed weight initialization method are applied to the three CNN models. All the models are trained on a learning rate 1e-4 with Adam optimizer. 

The figure below details the performances of the proposed weight initialization method on four public remote senging datasets, namely, UC-Merced, AID, KSA, and PatternNet.

<img src="doc/img1.png" />

The training progress plots in the figures below illustrate the performance of the proposed weight initialization method, as well as the Xavier, He, and zerO methods, on the CIFAR-100 dataset. The first figure displays the training progress of validation accuracy, while the second figure focuses on validation loss.

The analysis of the plots shows that the proposed weight initialization method outperforms the three other weight initialization techniques in terms of both accuracy and loss, as shown in both the overall training progress and the zoomed-in subplots. The performance advantage of the proposed method is visually apparent, with consistently higher accuracy values and lower loss values throughout the training process.

The comparison with He, Xavier, and zerO initialization methods further confirms the superior performance of the proposed approach. Notably, the zoomed-in subplots highlight the enhanced accuracy and reduced loss achieved by our proposed method in the final ten iterations. These findings highlight the effectiveness of the proposed weight initialization method in improving accuracy and minimizing the discrepancy between predicted and actual values.

<img src="doc/img2.png" />
<img src="doc/img3.png" />

### Dataset Setup
To download the dataset:
<ul>
  <li><a href="http://weegee.vision.ucmerced.edu/datasets/landuse.html">UC-Merced</a></li>
  <li><a href="https://drive.google.com/file/d/1H400Qamkl7oVCvvMzcQ72N0-jEZuegk5/view?usp=sharing">KSA</a></li>
  <li><a href="https://captain-whu.github.io/AID/">AID</a></li>
  <li><a href="https://sites.google.com/view/zhouwx/dataset">PatternNet</a></li>
</ul>
<br>
before running the code, you have to put the dataset as zipped files in compressed directory. The code will unzip it and split it randomly.
<pre>
 .
 └── data 
     └── compressed 
         ├── UCMerced_LandUse.zip 
         ├── KSA.zip 
         ├── AID.zip
         └── PatternNet.zip 
</pre>

### Help
running parameters
<table>
  <tr>
    <th>Parameter Name</th>
    <th>Description</th>
    <th>Default</th>
  </tr>
  <tr>
    <td>[-ds][--dataset_name]</td>
    <td>dataset name should be ucmerced, aid, ksa, patternnet, or wadii</td>
    <td>ucmerced</td>
  </tr>
  <tr>
    <td>[-mn][--model_name]</td>
    <td>model name should be any vgg, resnet, and mobilenet models</td>
    <td>mobilenet_v2</td>
  </tr>
  <tr>
    <td>[-wi][--weight_init]</td>
    <td>weight initialization method should be ether xavier or he to use the famouse method, and you can choose any other name to use our proposed method</td>
    <td>custom</td>
  </tr>
  <tr>
    <td>[-is][--image_size]</td>
    <td>image size tfor data transforms</td>
    <td>224</td>
  </tr>
  <tr>
    <td>[-tr][--train]</td>
    <td>training option</td>
    <td>False</td>
  </tr>
  <tr>
    <td>[-ev][--eval]</td>
    <td>evaluation option</td>
    <td>False</td>
  </tr>
  <tr>
    <td>[-evs][--eval_summary]</td>
    <td>evaluation summary option</td>
    <td>False</td>
  </tr>
  <tr>
    <td>[-ep][--epochs]</td>
    <td>training iteration number</td>
    <td>100</td>
  </tr>
  <tr>
    <td>[-bs][--batch_size]</td>
    <td>training/evaluation batch size</td>
    <td>16</td>
  </tr>
  <tr>
    <td>[-lr][--learning_rate]</td>
    <td>training learning rate</td>
    <td>0.0001</td>
  </tr>
  <tr>
    <td>[-sv][--save]</td>
    <td>save the model and training history</td>
    <td>True</td>
  </tr>
  <tr>
    <td>[-ow][--overwrite]</td>
    <td>overwrite the current model with the same dataset_name, model_name, and init_name</td>
    <td>True</td>
  </tr>
  <tr>
    <td>[-pr][--printing]</td>
    <td>print the used hyperparameters, dataset details, and model training progress</td>
    <td>True</td>
  </tr>
  <tr>
    <td>[-an][--avg_num]</td>
    <td>number of the evaluation for taking the average</td>
    <td>10</td>
  </tr>
  <tr>
    <td nowrap>[-sp][--summary_save_path]</td>
    <td>save path of the summary file</td>
    <td>./results/log/</td>
  </tr>
</table>


### Usage
To run the training and evaluation with the default values, run the following command:
```
python run.py --train --eval
```
To run the code using your own parameters, run the following command
```
!python run.py                \
--dataset_name ucmerced       \
--model_name mobilenet_v2     \
--weight_init proposed        \
--train                       \
--eval                        \
--eval_summary                \
--epochs 1                    \
--batch_size 32               \
--learning_rate 0.0001        \
--save                        \
--overwrite                   \
--printing                    \
--avg_num 3                   \
--summary_save_path summary
```
or run the following command for short:
```
!python run.py -ds ucmerced -mn mobilenet_v2 -wi proposed -tr -ev -evs -ep 1 -bs 32 -lr 0.0001 -sv -ow -pr -an 3 -sp summary
```

### Notes
<ul>
  <li>setting printing option as false will make you run the code in silent mode</li>
  <li>eval_summary option will evaluate all the saved checkpoints and generate an organized CSV file that contains all the evaluations</li>
  <li>in eval_summary option, the evaluations will use the validation set with avg_num. For example, I ran the code with eval_summary and avg_num = 3, this will evaluate the validation set 3 times and compute the average of all the the results</li>
  <li>setting overwrite option as false will stop the training if you had a saved checkpoint and history with the same parameters (dataset name, model name, weight init, and epochs)</li>
</ul>
