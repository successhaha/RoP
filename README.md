
# How to run

### RoP


```bash
python3 main_label_noise.py --gpu 0 --model 'PreActResNet18' --robust-learner 'SOP' -rc 0.9 -rb 0.1 \
          --dataset CIFAR10 --noise-type $noise_type --n-class 10 --lr-u 10 -se 10 --epochs 300 \
          --fraction $fraction --selection RoP --save-log True \
          --metric cossim --uncertainty LeastConfidence --tau 0.975 --eta 1 --balance True
```


