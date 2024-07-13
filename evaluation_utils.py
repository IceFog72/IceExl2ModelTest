# evaluation_utils.py

import torch

def evaluate_model(model, tokenizer, prompts, labels, answer_logits, progress_callback=None):
    score = 0.0
    model_outputs = []
    for prompt, label in zip(prompts, labels):
        prompt_ids = tokenizer.encode(prompt)[:, :-1]
        logits = model.forward(prompt_ids, last_id_only=True).float()
        logits_ans = logits[:, :, answer_logits]
        prob_ans = torch.softmax(logits_ans, dim=-1)
        predicted_label = torch.argmax(prob_ans).item()
        
        # Convert string label to integer if necessary
        if isinstance(label, str):
            label_index = ord(label) - ord('A')
        else:
            label_index = label
        
        score += prob_ans[0, 0, label_index]
        
        # Decode the predicted answer
        predicted_answer = chr(ord('A') + predicted_label)
        
        model_outputs.append({
            'prompt': prompt,
            'correct_label': chr(ord('A') + label_index) if isinstance(label, int) else label,
            'predicted_label': predicted_answer,
            'is_correct': predicted_label == label_index
        })
        
        # Call the progress callback if provided
        if progress_callback:
            progress_callback()
    
    return score / len(prompts), model_outputs