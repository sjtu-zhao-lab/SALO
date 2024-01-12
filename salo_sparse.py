import torch
import torch.nn.functional as F
import math

def matchingStatic_Block(attention_score: torch.Tensor,attention_mask_out, match_size, pe_size, global_nums, random_nums,dilation=1):
    with torch.no_grad():
        attention = attention_score.clone()
        attention_mask = attention_mask_out.clone()
        batch_size_o, head_o, row_o, col_o = attention.size()
        row_before, col_before = row_o, col_o
        if row_o % match_size != 0 or col_o % match_size != 0:
            add_or = match_size - (row_o % match_size)
            add_oc = match_size - (col_o % match_size)
            attention = attention.resize_(batch_size_o,head_o,row_o + add_or,col_o+ add_oc)
            batch_size_o, head_o, row_o, col_o = attention.size()
            attention_mask = attention_mask.resize_(batch_size_o,1,row_o,col_o)
        device = attention.device


        dataLen = (attention_mask==0)
        s = attention.split(match_size,dim=2)
        c = torch.cat(s,dim=0)
        ss = c.split(match_size,dim=3)
        attention = torch.cat(ss,dim=0)
        batch_size, head, row, col = attention.size()

        if dataLen.size()[2] != 1:
            dataLen = dataLen.view(batch_size_o,1,row_o,col_o).expand((batch_size_o,head_o,row_o,col_o))
        else:
            dataLen = dataLen.view(batch_size_o,1,1,col_o).expand((batch_size_o,head_o,row_o,col_o))
        dataLen_tran = dataLen.transpose(2,3)

        s = dataLen.split(match_size,dim=2)
        c = torch.cat(s,dim=0)
        ss = c.split(match_size,dim=3)
        dataLen_block = torch.cat(ss,dim=0) 

        s = dataLen_tran.split(match_size,dim=2)
        c = torch.cat(s,dim=0)
        ss = c.split(match_size,dim=3)
        dataLen_tran_block = torch.cat(ss,dim=0) 


        mask = torch.zeros_like(attention,dtype=torch.int32)
        
        if global_nums != 0:
            attentionMask_global_col = attention.sum(dim=2,keepdim=True)
            attentionMask_global_col, attentionMask_global_col_ind = attentionMask_global_col.topk(global_nums,dim=3, largest=True)
            attentionMask_global_col_ind = attentionMask_global_col_ind.reshape(-1,global_nums).t().reshape(-1)

            attentionMask_global_row = attention.sum(dim=3,keepdim=True)
            attentionMask_global_row, attentionMask_global_row_ind = attentionMask_global_row.topk(global_nums,dim=2, largest=True)
            attentionMask_global_row_ind = attentionMask_global_row_ind.reshape(-1,global_nums).t().reshape(-1)

            batch_ind = torch.arange(0,batch_size).repeat(head).reshape(head,batch_size).t().reshape(-1).repeat(global_nums).to(device)
            head_ind = torch.arange(0,head).repeat(batch_size).repeat(global_nums).to(device)
            mask[batch_ind,head_ind,:,(attentionMask_global_col_ind)] = 1
            mask[batch_ind,head_ind,(attentionMask_global_row_ind),:] = 1
            
        diag_sum = torch.zeros((batch_size, head, 2*row-1), device=device)

        for i in range(2*row-1):
            diag = torch.diagonal(attention, offset=(i-row+1), dim1=2, dim2=3)
            diag_sum[:, :, i] += diag.sum(dim=2)

        diag_b, diag_h, diag_col = diag_sum.size()
        filters = torch.ones((diag_h, 1, pe_size),device=device)
        window_sum = F.conv1d(diag_sum, filters,groups=diag_h,dilation=dilation)

        center = torch.argmax(window_sum,dim=2,keepdim=True).view(batch_size,head,1,1).expand((batch_size,head,row,row))
        diag_idx = torch.arange(0, row, device=device).expand((row, row))
        diag_idx = diag_idx.clone() + torch.arange(row-1,-1,-1, device=device).view((row,1)).expand((row,row))
        diag_idx = diag_idx.view(1,1,row,row).expand((batch_size,head,row,row))

        local = torch.logical_and(diag_idx >= center, diag_idx <= center+(dilation*(pe_size-1)))
        local = torch.logical_and(local, (diag_idx - center) % dilation == 0)

        local = torch.logical_and(dataLen_block, local)
        local = torch.logical_and(dataLen_tran_block, local)
        mask = mask | local

        attention[mask == 1] = 0
        if random_nums != 0:
            attention[~dataLen_block]=0
            random_v, random_ind = attention.topk(random_nums,dim=3,largest=True)
            random_ind = random_ind.reshape(-1,random_nums).t().reshape(-1)
            row_ind = torch.arange(0,row).reshape(1,1,-1).repeat(batch_size,head,random_nums).reshape(-1).to(device)
            batch_ind = torch.arange(0,batch_size).repeat(head*row).reshape(-1,batch_size).t().reshape(-1).repeat(random_nums).to(device)
            head_ind = torch.arange(0,head).reshape(1,-1).repeat(row,1).t().reshape(-1).repeat(batch_size*random_nums).to(device)
            mask[batch_ind,head_ind,row_ind,random_ind] = 1

        cc = mask.chunk(int(col_o/match_size),dim=0)
        cc = torch.cat(cc,dim=3)
        cc= cc.chunk(int(row_o/match_size),dim=0)
        mask = torch.cat(cc,dim=2)
        mask = mask & dataLen
        if row_before % match_size != 0 or col_before % match_size != 0:
            mask = mask.resize_(batch_size_o, head_o,row_before, col_before)
    return mask 
