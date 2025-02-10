import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseEvent
from PIL import Image
import base64
import io
import random



def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"



class BoundingBoxEnv(gym.Env):
    def __init__(self, image, gt_bbox):
        super(BoundingBoxEnv, self).__init__()
        self.image = image
        self.gt_bbox = gt_bbox

        
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32)

        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        
        self.agent_bbox = np.array([random.uniform(0, 1), random.uniform(0, 1), 0.1, 0.1])

    def reset(self):
        
        self.agent_bbox = np.array([random.uniform(0, 1), random.uniform(0, 1), 0.1, 0.1])
        return self.agent_bbox

    def step(self, action):
        
        self.agent_bbox += action
        self.agent_bbox = np.clip(self.agent_bbox, 0, 1)  

        
        reward = self._calculate_iou(self.agent_bbox, self.gt_bbox)

        
        done = reward > 0.8  

        return self.agent_bbox, reward, done, {}

    def _calculate_iou(self, box1, box2):
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        
        img_width, img_height = self.image.shape[1], self.image.shape[0]
        x1, y1, w1, h1 = x1 * img_width, y1 * img_height, w1 * img_width, h1 * img_height
        x2, y2, w2, h2 = x2 * img_width, y2 * img_height, w2 * img_width, h2 * img_height

        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area, box2_area = w1 * h1, w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def render(self):
        
        img = self.image.copy()
        h, w, _ = img.shape

        
        def denorm_bbox(bbox):
            x, y, bw, bh = bbox
            return int(x * w), int(y * h), int(bw * w), int(bh * h)

        
        gt_x, gt_y, gt_w, gt_h = denorm_bbox(self.gt_bbox)
        ag_x, ag_y, ag_w, ag_h = denorm_bbox(self.agent_bbox)

        
        cv2.rectangle(img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 255, 0), 5)
        cv2.rectangle(img, (ag_x, ag_y), (ag_x + ag_w, ag_y + ag_h), (255, 0, 0), 5)

        plt.imshow(img)
        plt.axis("off")
        plt.show()



def on_click(event: MouseEvent):
    global start_x, start_y, rect

    
    if event.inaxes:
        if start_x is None and start_y is None:
            
            start_x, start_y = event.xdata, event.ydata
        else:
            
            width = event.xdata - start_x
            height = event.ydata - start_y

            
            rect = patches.Rectangle((start_x, start_y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            fig.canvas.draw()



def on_close(event):
    global start_x, start_y, rect
    if start_x is not None and start_y is not None and rect is not None:
        
        print(f"Coordenadas da caixa delimitadora: ({start_x}, {start_y}) até ({start_x + rect.get_width()}, {start_y + rect.get_height()})")

        
        gt_bbox = np.array([start_x / image.shape[1], start_y / image.shape[0],
                            rect.get_width() / image.shape[1], rect.get_height() / image.shape[0]])

        
        env = BoundingBoxEnv(image, gt_bbox)

        
        state = env.reset()

        
        episode = 0

        while True:
            action = np.random.uniform(-0.05, 0.05, 4)  
            next_state, reward, done, _ = env.step(action)

            print(f"Episódio {episode}: Recompensa = {reward}")

            if done:
                print("Bounding box final encontrada!")
                env.render()
                break

            episode += 1



image_path = 'RL-in-images/Exemple_01.jpg'  
image_data = encode_image(image_path)
image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


start_x, start_y = None, None
rect = None


fig, ax = plt.subplots()
ax.imshow(image)


fig.canvas.mpl_connect('button_press_event', on_click)


fig.canvas.mpl_connect('close_event', on_close)


plt.show()
