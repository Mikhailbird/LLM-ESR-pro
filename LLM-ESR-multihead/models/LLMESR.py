# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMESR_SASRec(DualLLMSASRec): 
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

        self.alpha = args.alpha
        self.theta = args.theta
        self.theta_long = args.theta_long
        self.theta_short = args.theta_short
        self.item_num = item_num
        self.args = args
        self.dev = device

        self.loss_func = nn.BCEWithLogitsLoss()

        # align loss
        if args.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif args.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError("Unknown user_sim_func")

        self.projector1 = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)
        self.projector2 = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)

        # item-level alignment (optional)
        if args.item_reg:
            self.item_reg = True
            self.beta = args.beta
            self.reg = Contrastive_Loss2()
        else:
            self.item_reg = False

        # tail-aware prediction heads
        self.long_tail_head = nn.Linear(args.hidden_size, item_num)
        self.short_tail_head = nn.Linear(args.hidden_size, item_num)

        self._init_weights()

    def forward(self, seq, pos, neg, positions, user_type=None, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # current user embedding
        log_feats = self.log2feats(seq, positions)[:, -1, :]  # (B, hidden)

        # alignment with similar users
        sim_seq = kwargs["sim_seq"].view(-1, seq.shape[1])
        sim_pos = kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_feats = self.log2feats(sim_seq, sim_pos)[:, -1, :].detach().view(seq.shape[0], sim_num, -1)

        if self.args.ablation_mode == "mean":
            sim_repr = sim_feats.mean(dim=1)
        elif self.args.ablation_mode == "top1":
            sim_repr = sim_feats[:, 0, :]
        else:
            raise ValueError(f"Unknown ablation_mode: {self.args.ablation_mode}")

        align_loss = self.align(log_feats, sim_repr)
        loss += self.alpha * align_loss

        # item embedding alignment loss
        if self.item_reg:
            unfold_item = torch.masked_select(seq, seq > 0)
            llm_item = self.adapter(self.llm_item_emb(unfold_item))
            id_item = self.id_item_emb(unfold_item)
            loss += self.beta * self.reg(llm_item, id_item)

        # user-type-specific prediction heads
        if user_type is not None:
            long_mask = (user_type == 1)
            short_mask = (user_type == 0)
            target = pos[:, 0]

            if long_mask.any():
                logits = self.long_tail_head(log_feats[long_mask])  # [B, item_num]
                labels = torch.zeros_like(logits)
                labels.scatter_(1, target[long_mask].unsqueeze(1), 1)
                loss += self.theta * self.loss_func(logits, labels)

            if short_mask.any():
                logits = self.short_tail_head(log_feats[short_mask])  # [B, item_num]
                labels = torch.zeros_like(logits)
                labels.scatter_(1, target[short_mask].unsqueeze(1), 1)
                loss += (1 - self.theta) * self.loss_func(logits, labels)

        return loss

    def predict(self, seq, item_indices, positions, user_type=None, **kwargs):
        log_feats = self.log2feats(seq, positions)
        user_feat = log_feats[:, -1, :]  # (B, H)
        item_embs = self._get_embedding(item_indices)  # (B, I, H)
        dot_scores = item_embs.matmul(user_feat.unsqueeze(-1)).squeeze(-1)

        if user_type is None:
            return dot_scores

        logits = dot_scores.clone()
        long_mask = (user_type == 1)
        short_mask = (user_type == 0)

        if long_mask.any():
            long_feat = user_feat[long_mask]
            long_logits_all = self.long_tail_head(long_feat)
            long_logits = torch.gather(long_logits_all, 1, item_indices[long_mask])
            logits[long_mask] = (1 - self.theta_long) * dot_scores[long_mask] + self.theta_long * long_logits

        if short_mask.any():
            short_feat = user_feat[short_mask]
            short_logits_all = self.short_tail_head(short_feat)
            short_logits = torch.gather(short_logits_all, 1, item_indices[short_mask])
            logits[short_mask] = (1 - self.theta_short) * dot_scores[short_mask] + self.theta_short * short_logits

        return logits



class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        loss += self.alpha * align_loss

        return loss



class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)  # get the original loss
        
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        if self.user_sim_func == "cl":
            # align_loss = self.align(self.projector1(log_feats), self.projector2(sim_log_feats))
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        loss += self.alpha * align_loss

        return loss



