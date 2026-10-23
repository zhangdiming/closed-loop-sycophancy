import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2Oracle:
    can_generate = False

    def __init__(self, repo="google/owlv2-large-patch14-ensemble", device="cuda", threshold=0.3, nms_iou=0.5):
        self.proc = Owlv2Processor.from_pretrained(repo)
        self.model = Owlv2ForObjectDetection.from_pretrained(repo).to(device).eval()
        self.device, self.thr, self.nms_iou = device, threshold, nms_iou

    @torch.no_grad()
    def detect(self, image, classes):
        texts = [[f"a photo of a {c}" for c in classes]]
        inputs = self.proc(text=texts, images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        w, h = image.size; s = max(w, h)
        res = self.proc.post_process_object_detection(out, threshold=self.thr, target_sizes=torch.tensor([[s, s]]))[0]
        boxes, scores, labels = res["boxes"].cpu(), res["scores"].cpu(), res["labels"].cpu()
        dets = {c: [] for c in classes}
        for ci, c in enumerate(classes):
            m = labels == ci
            if m.any():
                from torchvision.ops import nms
                keep = nms(boxes[m], scores[m], self.nms_iou)
                dets[c] = [(boxes[m][k].tolist(), float(scores[m][k])) for k in keep]
        return dets

    def judge_fact(self, image, fact):
        t, a = fact["type"], fact["args"]
        if t == "color": return None
        classes = [a["class"]] + ([a["other"]] if t == "position" else [])
        dets = self.detect(image.convert("RGB"), classes)
        n = len(dets[a["class"]])
        if t == "presence": ans = n >= 1
        elif t == "count": ans = n == a["count"]
        else:
            if n == 0 or len(dets[a["other"]]) == 0: ans = False
            else:
                (x1, y1, x2, y2), _ = max(dets[a["class"]], key=lambda d: d[1]); (u1, v1, u2, v2), _ = max(dets[a["other"]], key=lambda d: d[1])
                cx, cy, ox, oy = (x1 + x2) / 2, (y1 + y2) / 2, (u1 + u2) / 2, (v1 + v2) / 2
                rel = a["relation"]
                ans = {"left of": cx < ox, "right of": cx > ox, "above": cy < oy, "below": cy > oy}[rel]
        conf = max([s for _, s in dets[a["class"]]], default=0.0)
        return {"answer": "yes" if ans else "no", "p_yes": None, "p_no": None, "mass": None,
                "raw": f"n={n} conf={conf:.2f}" + (f" n_other={len(dets[a['other']])}" if t == "position" else "")}
