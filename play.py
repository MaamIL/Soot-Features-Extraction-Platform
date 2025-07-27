from main import Config

import hiddenlayer as hl
import torch
from Mymodels.CNNencdec import CNNencdec
config = Config()
model = CNNencdec(config)
checkpoint = torch.load("C:/Users/User/Documents/Sooth_Features_Extraction_plat/Outputs_CNNencdec_both_2025-0527-111352/best_flame_model.pth", map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
from torchviz import make_dot

# Forward pass once to generate graph
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)

# Visualize
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")
# dummy_input = torch.randn(1, 3, 224, 224)
# hl_graph = hl.build_graph(model, dummy_input)
# hl_graph.save("model_graph", format="png")