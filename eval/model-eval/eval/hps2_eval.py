import hpsv2

is_rewrite ="short"
img_list = ['xxx']
for img_path in img_list:
    print(f"=========={img_path}==========")
    hpsv2.evaluate(f"/model-eval/{is_rewrite}-img/{img_path}/", f"/model-eval/{is_rewrite}-img/banchmark") 
