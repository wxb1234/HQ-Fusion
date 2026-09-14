import torch 
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    # return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
    """
    items包含批次数量的图及gt
    如果 len(items[0]) == 3 说明包含 vis,ir,target
    """
    if items and len(items[0]) == 3:
        '''返回字典格式，更清晰明了'''
        visible_imgs = torch.cat([x[0][None] for x in items], dim=0)
        infrared_imgs = torch.cat([x[1][None] for x in items], dim=0)
        targets = [x[2] for x in items]

        # return {
        #     'vis': visible_imgs,
        #     'ir': infrared_imgs,
        #     'targets': targets
        # }
        return {
            # 'vis': visible_imgs,
            'fusion_image': torch.cat((infrared_imgs, visible_imgs), dim=1),
            'targets': targets
        }
    else:
        return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
