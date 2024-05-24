import torch
import vgpu


# custom_device = torch.device("vgpu")
custom_device = "vgpu:0"

print("\n\n[PY DEBUG] tst_1 = torch.tensor(0.0, device=custom_device)")
tst_1 = torch.tensor(0.0, device=custom_device)

print(
    "\n\n[PY DEBUG] a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=custom_device)"
)
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=custom_device)

print("\n\n[PY DEBUG] a_cpu = a.cpu()")
a_cpu = a.cpu()

print(
    "\n\n[PY DEBUG] b = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 10.0, 11.0, 12.0]], device=custom_device)"
)
b = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [8.0, 10.0, 11.0, 12.0]],
    device=custom_device,
)

print("\n\n[PY DEBUG] b_cpu = b.cpu()")
b_cpu = b.cpu()

print("\n\n[PY DEBUG] c = torch.mm(a, b)")
c = torch.mm(a, b)

print("\n\n[PY DEBUG] c_cpu = c.cpu()")
c_cpu = c.cpu()

print("\n\n[PY DEBUG] c_cpu_check = torch.mm(a_cpu, b_cpu)")
c_cpu_check = torch.mm(a_cpu, b_cpu)

print("\n\n[PY DEBUG] av = a.view(6)")
av = a.view(6)

print("\n\n[PY DEBUG] av_cpu = av.cpu()")
av_cpu = av.cpu()


# print("\n\n[PY DEBUG] a[0]")
# a[0]

# print("\n\n[PY DEBUG] print(av)")
# print(av)

# print(c)
