import transfomers
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM

class CustomTrainerDistill(transformers.Trainer):
    def __init__(self, *args, teacher_model=None, teacher_input = None, lambda_param = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.lambda_param = lambda_param
        self.teacher_input = teacher_input
        self.devce = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(self.devce)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        student_output = model(**inputs)
        student_logits = student_output.get("logits")

        ti = transformers.BatchEncoding(process_inp_att(self.teacher_input['input_ids'], self.teacher_input['attention_mask'], self.devce))

        with torch.no_grad():
          teacher_output = self.teacher(**ti)
          teacher_logits = teacher_output.get("logits")

        input_ids = inputs['input_ids']
        batch_size, sequence_length = input_ids.shape[:2]


        # Compute the distillation loss
        soft_teacher = nn.functional.softmax(teacher_logits, dim=-1)
        soft_student = nn.functional.log_softmax(student_logits, dim=-1)

        #print(soft_teacher.shape, soft_student.shape)
        #print(teacher_logits.shape, student_logits.shape)

        dist_loss_fct = nn.KLDivLoss(reduction = 'batchmean')
        dist_loss = dist_loss_fct(soft_student, soft_teacher)

        # Compute the student loss
        stu_loss_fct = nn.CrossEntropyLoss()
        stu_loss = stu_loss_fct(student_logits.view(-1, num_labels), labels.view(-1))

        # Calculate final loss
        loss = (1. - self.lambda_param) * stu_loss + self.lambda_param * dist_loss
        return (loss, student_output) if return_outputs else loss