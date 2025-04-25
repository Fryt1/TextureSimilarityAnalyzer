<h2 id="jOVZ6">贴图相似性检测核心算法</h2>
<h3 id="oL0Te">一、三种算法的特性与互补性</h3>

1. **pHash（感知哈希）**  
   • **优势**：对缩放、轻微颜色调整和格式转换（如JPEG压缩）具有较强适应性，计算速度快，适合快速去重。  
   • **劣势**：对旋转、裁剪、局部遮挡敏感，且无法捕捉高级语义特征。

参考文章:[图片相似度识别：pHash算法](https://mp.weixin.qq.com/s?__biz=MzAwNTIyMDU3NA==&mid=2648492655&idx=1&sn=9ffa69ff3b83ab7bf73cee0f0fe21386&chksm=83379bdeb44012c8d96be91f2032126272aabe7480c0f830b416faa8dcc160ca878454249a8d&token=783893672&lang=zh_CN#rd)

2. **SIFT（尺度不变特征变换）**  
   • **优势**：对旋转、缩放、视角变化等几何变换高度鲁棒，能通过关键点匹配识别局部特征差异。  
   • **劣势**：计算复杂度高，对全局颜色变化（如色调调整）和均匀噪声（如高斯噪声）敏感。

参考文章:[SIFT特征的提取过程](https://zhuanlan.zhihu.com/p/445681832)

3. **直方图算法**  
   • **优势**：对颜色分布变化敏感（如整体调色），适合基于颜色的相似性检测，计算简单。  
   • **劣势**：无法区分结构差异，对局部颜色调整（如添加水印）和几何变换无效。

参考文章:[python图像识别---------图片相似度计算](https://zhuanlan.zhihu.com/p/68215900)

****

**合使用的可能性**：三者可分别从哈希特征（全局结构）、关键点（局部几何）和颜色分布（全局色彩）三个维度互补。例如：

---

<h3 id="GdyDu">二、对常见干扰的适应性对比</h3>

| **干扰类型**      | **pHash**    | **SIFT**         | **直方图算法**   |
| ----------------- | ------------ | ---------------- | ---------------- |
| **颜色调整**      | 鲁棒（轻微） | 敏感（依赖纹理） | 高度敏感         |
| **缩放**          | 鲁棒         | 鲁棒             | 敏感（需归一化） |
| **旋转**          | 敏感         | 鲁棒             | 敏感             |
| **添加噪声**      | 中等敏感     | 鲁棒（稀疏噪声） | 敏感             |
| **局部遮挡/水印** | 敏感         | 鲁棒（部分匹配） | 敏感             |



<h3 id="QGX6m">三、由上述敏感性和鲁棒性有Combined方案</h3>

由于**SIFT**的性能相比**pHash**和**直方图算法**比较差，并且**pHash**和**直方图算法无法处理水印这种局部结构变化，所以使用分层判断方案**

<h4 id="g3nEB">第一步:使用pHash和直方图算法进行结构/全局颜色的粗粒度筛选</h4>

**pHash**对缩放、轻微颜色调整和格式转换（如JPEG压缩）具有较强适应性，但是对于极端的颜色调整不够敏感，而**直方图**对颜色分布变化敏感（如整体调色），这两个可以筛选出大部分相似图片。

```python
def _coarse_compare(self, base_features, target_features, threshold):
        phash_sim = self.pHash_similarity(base_features['phash'], target_features['phash'])
        hist_sim = 1 - cv2.compareHist(base_features['hist'], target_features['hist'], cv2.HISTCMP_BHATTACHARYYA)
        
        # 阶段1：颜色特征验证
        if(hist_sim<0.75):
            return False
        
        # 阶段2：结构特征交叉验证
        phash_weight = 0.6 if phash_sim > 0.85 else 0.4
        combined_score = phash_weight * phash_sim + 0.6 * hist_sim
            
        return combined_score > threshold
```

<h5 id="ClKUW">TODO:在combined模式下新增参数手动控制直方图/Phash的阈值以应对更加多种的图片种类</h5>




<h4 id="ifvkT">第二步:使用SIFT对粗粒度筛选掉的中被误筛选掉的(局部遮挡/水印/添加噪声)和粗粒度筛选出来相似的图片再进行相似度对比，补上被误筛选掉的相似图。</h4>

**pHash**和**直方图**对于噪声和遮挡和水印太敏感，会导致在第一步**pHash和直方图**进行粗粒度筛选的时候会漏掉遮挡和水印的情况，所以使用**SIFT**对第一步粗粒度筛选误筛选掉的纹理再进行筛选，将粗粒度被筛选掉的纹理与粗粒度已经筛选出来的相似图片再SIFT进行相似性对比。得到细粒度对比的最终结果。



**三者结合能覆盖颜色、结构、局部特征等多维度信息，理论上鲁棒性优于单一算法。**

****

<h5 id="wx0Hu">TODO:1.优化效率sift算法效率</h5>




<h2 id="WudZH">UI界面设计</h2>

提供<font style="color:rgb(0,0,0);">查看、筛选和调整匹配结果功能</font>

<font style="color:rgb(0,0,0);">使用</font><font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(243, 243, 243);">PyQt开发，拥有导入图片，检测相似性并分组，批量删除/改名/移动，切换PHash/Sift/直方图/combined算法的功能</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40675728/1745570826511-abbb4f7d-c244-4238-809d-250422211281.png)

![](https://cdn.nlark.com/yuque/0/2025/png/40675728/1745585001968-52cdb597-a1cf-4489-9337-82bd9b12c8e3.png)

<h5 id="km3Di">TODO:1.手动调整匹配结果 </h5>


![](https://cdn.nlark.com/yuque/0/2025/png/40675728/1745586609263-13dfae2c-0d91-4871-bc8e-7b7413720c8a.png)

算法选择:

- [x] 1.pHash(以图片结构判断图片相似)
- [x] 2.SIFT(匹配局部结构特征，可以处理水印这类情况)
- [x] 3.Histogram(依赖颜色分布的图片匹配算法)
- [x] 4.Combined(组合pHash，SIFT，Histogram三种算法，以适应复杂情况)



<h2 id="RZdcs">已知Bug</h2>

**1.非第一次导入图片会导致死锁**

