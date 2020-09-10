from u2net_test import U_2net
# argument로 주소 받을거임
path = 'data/ex'

net = U_2net.getNet()
img_list = U_2net.getData(path)
loader = U_2net.getLoader(img_list)
U_2net.run(img_list, loader, net, path)

# U_2net.getRoot()

# net = U_2net.getNet() #1
# img_list =U_2net.getRoot() #2
# loader = U_2net.getLoader(img_list) #3


# run command
# U_2net.run(img_list,loader,net)
