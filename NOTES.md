# Notes

## TODO

- one good train

## QUESTIONS

- should you load weights into the models?
- onsite test data loader?

## Options


`--backbone`            // backbone to use [resnet18*|efficientnet]
`--scratch`             // train from scratch (will set full_ft to True)
`--full-ft`             // do full fine tuning

`--run-name`            // 
`--mode`                // [train*|test|predict]

`--epochs`
`--lr`
`--batch-size`

`--loss-fn`             // 
`--attention`           // Attention mechanism to use [SE|MHA]






classifier only vs. full fine-tuning
FROM SCRATCH vs. PRETRAINED BACKBONE 

default:
- load pretrained backbone
- classifier only


OPTIONS:
- train
- test (offsite test set)
- predict (onsite test set, produce csv)


## MODEL LOADING

Options:
- backbone          `resnet | efficientnet`
- pretrained params `scratch | pretrained_backbone | finetuned`
- param unfreezing  `classifier_only | all`
