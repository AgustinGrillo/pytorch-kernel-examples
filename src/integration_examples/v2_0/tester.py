import torch
import vgpu


# custom_device = torch.device("vgpu")
custom_device = "vgpu:0"

tst_1 = torch.tensor(0, device=custom_device)
tst_2 = torch.tensor([0], device=custom_device)
tst_3 = torch.tensor([0, 0], device=custom_device)

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=custom_device)
a_cpu = a.cpu()
b = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 10.0, 11.0, 12.0]],
    device=custom_device,
)
b_cpu = b.cpu()

c = torch.mm(a, b)
c_cpu = c.cpu()
c_cpu_check = torch.mm(a_cpu, b_cpu)

av_cpu = a.view(6).cpu()

# print(c)
