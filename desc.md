## Model Name:
ALBERT (ensemble model)

## Affiliation:
Zhiyan Technology

## Model Description: 
We only used the question and choices without any extra data. 
For one sample, We concatenated them such as `<cls> question_tokens <sep> choice_tokens <sep> `for every choice. And using `mutlichoice` train mode.
`question_tokens = 'Q: ' + question`，
`choice_tokens = 'A: ' + choice`
```
def forward(input_ids):
   """
   input_ids: (None, 5, MAX_SEQ_LENGTH)
   """
   input_ids = input_ids.reshape(-1, input_ids.size(-1))
   outputs = ALBERT(inputs)
   pool_output = sequence_summary(outputs)  # (5 * None, hidden_size)
   cls_output = classifier(pool_output)  # (None, 5)
   return cls_output
```


## Experiment Details: 
In our experiments, we used the pre-trained ALBERT-xxlarge-v2 model from https://github.com/google-research/ALBERT. The accuracy is 83.7%/76.5% on the dev/test dataset. And the single model's accuracy is 80.9%/80.0%/80.5%/81.2%/80.4% on the dev dataset(using 5 different seed). The parameters are listed as below:
- `sequence_summary function` concatenate the last 4 layers of ALBERT
- `classifier function` use a fc-layer
- MAX_SEQ_LENGTH = 80
- TRAIN_BATCH_SIZE = 4
- GRADIENT_ACCUMULATION_STEPS = 4
- LEARNING_RATE = 1e-5
- WEIGHT_DECAY = 0.0
- ADAM_EPSILON = 1e-8
- MAX_GRAD_NORM = 1.0
- NUM_TRAIN_STEPS = 2000
- WARMUP_STEPS = 608
- LOGGING_STEPS = 60
