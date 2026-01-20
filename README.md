
To install requirements:

           pip install -r requirements.txt


# How to run

### RoP


```bash
python3 main_label_noise.py --gpu 0 --model 'PreActResNet18' --robust-learner 'SOP' -rc 0.9 -rb 0.1 \
          --dataset CIFAR10 --noise-type $noise_type --n-class 10 --lr-u 10 -se 10 --epochs 300 \
          --fraction $fraction --selection RoP --save-log True \
          --metric cossim --uncertainty LeastConfidence --tau 0.975 --eta 1 --balance True
```

### Data Pruning Baselines: Uniform, SmallLoss, Margin, Forgetting, GraNd, Moderate, etc

Basically, the script is similar to that of RoP. For example, 

```bash
python3 main_label_noise.py --gpu 0 --model 'PreActResNet18' --robust-learner 'SOP' -rc 0.9 -rb 0.1 \
          --dataset CIFAR10 --noise-type $noise_type --n-class 10 --lr-u 10 -se 10 --epochs 300 \
          --fraction $fraction --selection *$pruning_algorithm* --save-log True \
```



