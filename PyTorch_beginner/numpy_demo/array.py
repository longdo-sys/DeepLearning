import numpy as np

data = np.array([[1, 2, 3], [2, 3, 4]])
print(data.dtype.name)

# 转换数据类型

float_data = np.array([1.5, 2.5, 3.5])
int_data = float_data.astype(int)
print(int_data)

# 广播机制
arr1 = np.array([[0], [1], [2], [3]])
arr2 = np.array([2, 3])
arr3 = arr1 + arr2
print(arr3)

# 数组与标量间的运算，分散到数组中每个数字


# #### ndarray 的索引和切片

# 创建
arr = np.arange(8)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 循环创建
arr_demo = np.empty((4, 4))
for i in range(4):
    arr_demo[i] = np.arange(i, i+4)

print(arr_demo)
print(arr_demo[[1, 3], [1, 2]])

# 布尔型索引

# dimension is 4 but corresponding boolean dimension is 3 看来boolean index和 数组的dimension匹配
student_name = np.array(['Tom', 'Lily', 'Jack', 'Rose'])

student_score = np.array([[79, 88, 80], [89, 90, 92], [83, 78, 85], [78, 76, 80]])

print(student_score[student_name == 'Rose'])
print(student_score[student_name == 'Rose', :-1])


# 数组的转置和轴对称

arr = np.arange(12).reshape(3, 4)

arr.T # 使用T属性对数组进行转置

arr = np.arange(16).reshape((2, 2, 4))

# 使用 transpose转置

arr.transpose(1, 2, 0)

# 使用swapaxes方法进行转置
arr.swapaxes(1, 0)
'''
结果 有点儿不明白
array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],

       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
'''

'''
通用函数：
np.sqrt(arr)#开方     np.abs(arr)#求绝对值    np.square(arr)#求平方
np.add(x, y)      # 计算两个数组的和
np.multiply(x, y) # 计算两个数组的乘积
np.maximum(x, y)  # 两个数组元素级最大值的比较
np.greater(x, y)  # 执行元素级的比较操作

'''


'''
数据处理
'''
# np.where可用于三目运算，也可用于返回断言成立时的索引 如：array([1, 2, 3, 1, 2, 3, 1, 2, 3])  idx = np.where(a > 2) 则idx 为(array([2, 5, 8], dtype=int32),)
# 栗子：假设有一个随机数生成的矩阵，希望将所有正值替换为2，负值替换为-2 np.where(arr>0,2,-2)

# 数组统计运算
arr.sum()
arr.mean()
arr.min()
arr.max()
# 最小值索引
arr.argmin()  # 同样有 arr.argmax()

# 计算元素累计和
arr.cumsum()

# 元素累计积
arr.cumprod()

# 取整 向左 np.floor() 以及 np.around(), 向右np.ceil()

# ###数组排序 arr.sort()数组同维度排序，arr.sort(0) 沿着编号为0的轴对元素排序

# ###检索
# np.any(arr > 0) arr的所有元素是否有一个大于0 np.all(arr > 0) arr的所有元素是否都大于0

# ### 唯一化(好像能去重)及其他集合逻辑 np.unique()  np.inld(arr, [11, 12]) 11, 12在arr中出现时其下标标成true

# ### 线性代数模块：arr_x.dot(arr_y) == np.dot(arr_x, arr_y)

# ### 随机数模块
print(np.random.rand(3, 3))
np.random.seed(0)
print(np.random.rand(5))
np.random.seed(0)
print(np.random.rand(5))

np.random.seed()
print(np.random.rand(5))

arr = np.random.rand(5)

arr = np.where(arr < 0.5, 0, 1)
print("arr:", arr)

print(arr.shape)
