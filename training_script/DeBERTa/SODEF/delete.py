from model_utils import get_a_phase1_model
import torch 
from train_utils import check_max_row_dist_matrix 

dummy = get_a_phase1_model(1024, 64, 2)

saved_temp = torch.load('D:/Pose/DeBERTa-sst2/phase2_last_ckpt.pth')
saved_temp[list(saved_temp.keys())[0]]

a, b = dummy.load_state_dict(saved_temp['model'], strict=False)
print(a, b)
print(dummy.fc.fc0.weight.shape)
check_max_row_dist_matrix(dummy.fc.fc0.weight.data.cpu().numpy().T, 2)