import cv2
import mediapipe as mp
import time
import numpy as np
import math
import ast
import operator as op

# ---------- Safe eval (only arithmetic) ----------
# Supports + - * / ** % and parentheses and unary +/-
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def safe_eval(expr: str):
    """
    Safely evaluate simple arithmetic expressions using ast.
    Raises ValueError on invalid expression.
    """
    try:
        parsed = ast.parse(expr, mode='eval')
        return _eval_ast(parsed.body)
    except Exception:
        raise ValueError("Invalid expression")

def _eval_ast(node):
    if isinstance(node, ast.Num):  # < Python 3.8
        return node.n
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Invalid constant")
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError("Operator not allowed")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError("Unary operator not allowed")
    raise ValueError("Unsupported expression")

# ---------- UI / Buttons ----------
class Button:
    def __init__(self, x, y, w, h, label, color_idx=0):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = str(label)
        self.color_idx = color_idx

    def draw(self, img, themes, hover=False):
        theme = themes[self.color_idx]
        bg = theme['button_hover'] if hover else theme['button']
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), theme['border'], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.label, font, 1.1, 2)[0]
        text_x = self.x + (self.w - text_size[0]) // 2
        text_y = self.y + (self.h + text_size[1]) // 2
        cv2.putText(img, self.label, (text_x, text_y), font, 1.1, theme['text'], 2, cv2.LINE_AA)

    def is_hover(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

# ---------- MediaPipe Helper ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_landmark_positions(hand_landmarks, img_w, img_h):
    lm = {}
    for id, lm_point in enumerate(hand_landmarks.landmark):
        lm[id] = (int(lm_point.x * img_w), int(lm_point.y * img_h))
    return lm

# ---------- Themes ----------
THEMES = [
    { 'bg': (18, 24, 34), 'button': (42, 95, 155), 'button_hover': (60, 120, 200), 'text': (255,255,255), 'border': (200,200,200) },
    { 'bg': (245,245,245), 'button': (50, 50, 50), 'button_hover': (80, 80, 80), 'text': (255,255,255), 'border': (10,10,10) },
    { 'bg': (20, 40, 20), 'button': (40, 150, 60), 'button_hover': (70, 190, 90), 'text': (255,255,255), 'border': (200,200,200) },
]

# ---------- Main app ----------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    btns = []
    rows = [
        ['7','8','9','/'],
        ['4','5','6','*'],
        ['1','2','3','-'],
        ['0','.','=','+'],
        ['C','(',')','%']
    ]
    btn_w, btn_h = 140, 100
    start_x, start_y = 40, 180
    gap = 10
    for r_idx, row in enumerate(rows):
        for c_idx, label in enumerate(row):
            x = start_x + c_idx * (btn_w + gap)
            y = start_y + r_idx * (btn_h + gap)
            btns.append(Button(x, y, btn_w, btn_h, label, color_idx=0))

    current_theme = 0
    expression = ""
    last_click_time = 0
    click_cooldown = 0.35
    five_finger_last = 0
    theme_cooldown = 1.0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        theme = THEMES[current_theme]
        overlay = np.full_like(frame, theme['bg'])
        frame = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        cursor_x, cursor_y = None, None
        pinch_distance = None
        five_fingers_up = False
        hovered = None

        if res.multi_hand_landmarks:
            hand_landmarks = res.multi_hand_landmarks[0]
            lm = get_landmark_positions(hand_landmarks, w, h)
            if 8 in lm:
                cursor_x, cursor_y = lm[8]
                cv2.circle(frame, (cursor_x, cursor_y), 8, (255, 50, 50), -1)
            if 4 in lm and 8 in lm:
                dx = lm[4][0] - lm[8][0]
                dy = lm[4][1] - lm[8][1]
                pinch_distance = math.hypot(dx, dy)
            tips = [4,8,12,16,20]
            pip_ids = {8:6,12:10,16:14,20:18}
            fingers_up = 0
            for tip in [8,12,16,20]:
                if tip in lm and pip_ids[tip] in lm:
                    if lm[tip][1] < lm[pip_ids[tip]][1]:
                        fingers_up += 1
            if 4 in lm and 3 in lm:
                if lm[4][0] < lm[3][0]:
                    fingers_up += 1
            five_fingers_up = (fingers_up == 5)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        disp_x, disp_y = start_x, 40
        disp_w = 4*(btn_w + gap) - gap
        disp_h = 100
        cv2.rectangle(frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), theme['button'], -1)
        cv2.rectangle(frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), theme['border'], 2)
        expr_text = expression if len(expression) < 30 else expression[-30:]
        cv2.putText(frame, expr_text, (disp_x + 15, disp_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, theme['text'], 3, cv2.LINE_AA)

        for b in btns:
            hover = False
            if cursor_x is not None and cursor_y is not None and b.is_hover(cursor_x, cursor_y):
                hover = True
                hovered = b
            b.draw(frame, THEMES, hover=hover)

        current_time = time.time()
        if five_fingers_up and (current_time - five_finger_last) > theme_cooldown:
            current_theme = (current_theme + 1) % len(THEMES)
            for b in btns:
                b.color_idx = current_theme
            five_finger_last = current_time

        clicked_button = None
        if pinch_distance is not None and pinch_distance < 40 and hovered is not None:
            if (current_time - last_click_time) > click_cooldown:
                clicked_button = hovered
                last_click_time = current_time

        if clicked_button:
            label = clicked_button.label
            if label == 'C':
                expression = ""
            elif label == '=':
                try:
                    result = safe_eval(expression)
                    expression = str(result)
                except Exception:
                    expression = "Error"
            else:
                expression += label

        if hovered:
            cv2.putText(frame, f"Hover: {hovered.label}", (start_x, start_y + 5*(btn_h+gap)+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220,220,220), 2, cv2.LINE_AA)

        info_lines = [
            "Instructions:",
            "- Point with index finger (cursor).",
            "- Pinch (index + thumb) to click.",
            "- Show all five fingers to change theme.",
            "- 'C' = clear, '=' = evaluate"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (disp_x + disp_w + 30, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Calculator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
