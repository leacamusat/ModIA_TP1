import argparse
import gradio as gr
import torch
from mnist_net import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recognize_digit(image):
    image = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.  # add a batch dimension
    prediction = model(image)[0]
    probabilities = torch.nn.functional.softmax(prediction, dim=0)
    return {str(i): probabilities[i].item() for i in range(10)}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default ="./mnist_net.pth", help='path of the weights')
    args = parser.parse_args()
    weights_path=args.weights_path
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.eval()

    gr.Interface(fn=recognize_digit, 
            inputs="sketchpad", 
            outputs=gr.outputs.Label(num_top_classes=3),
            live=True,
            description="Draw a number on the sketchpad to see the model's prediction.",
            ).launch(debug=True, share=True);
