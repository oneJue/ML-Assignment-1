<div align="center">

#  Assignment 1: 年龄预测回归任务

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Deadline](https://img.shields.io/badge/Deadline-Nov%202-red.svg)](http://172.23.166.133:3000)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

**📅 截止日期：11月2日** | **🏆 [查看排行榜](http://101.132.193.95:3000)**

**🎯实践平台** | [https://www.shuishan.net.cn/workshop/content?id=200](https://www.shuishan.net.cn/workshop/content?id=200)

---

</div>

## 📋 任务概述

> 在train.csv进行年龄预测回归任务，基于客户的职业、教育、信贷等特征(C列到Q列)预测客户年龄(B列)。

### 🎓 作业要求

```
✨ 1. 使用 SHAP (SHapley Additive exPlanations) 方法分析各特征对年龄预测的贡献度
✨ 2. 基于SHAP分析结果实施特征工程，优化模型性能
✨ 3. 独立实现机器学习模型的核心算法逻辑，不得调用任何外部机器学习库
```

<details>
<summary>❌ 禁止使用的库</summary>

- sklearn / scikit-learn
- tensorflow
- torch / pytorch
- keras
- xgboost
- lightgbm
- catboost
- statsmodels

</details>

---

## 📊 字段说明

<table>
<tr>
<td width="100%">

age: Age of the client (numeric)

- job: Type of job (categorical: "admin.", "blue-collar", "entrepreneur", etc.)
- marital: Marital status (categorical: "married", "single", "divorced")
- education: Level of education (categorical: "primary", "secondary", "tertiary", "unknown")
- default: Has credit in default? (categorical: "yes", "no")
- balance: Average yearly balance in euros (numeric)
- housing: Has a housing loan? (categorical: "yes", "no")
- loan: Has a personal loan? (categorical: "yes", "no")
- contact: Type of communication contact (categorical: "unknown", "telephone", "cellular")
- day: Last contact day of the month (numeric, 1-31)
- month: Last contact month of the year (categorical: "jan", "feb", "mar", …, "dec")
- duration: Last contact duration in seconds (numeric)
- campaign: Number of contacts performed during this campaign (numeric)
- pdays: Number of days since the client was last contacted from a previous campaign (numeric; -1 means the client was not previously contacted)
- previous: Number of contacts performed before this campaign (numeric)
- poutcome: Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")
- y: The target variable, whether the client subscribed to a term deposit (binary: "yes", "no")

</td>
</tr>
</table>

> 💡 更多信息：[Bank Marketing Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full)

---

## 📈 评测指标和评分方式

### 评测指标

- **MAE (Mean Absolute Error)**: 平均绝对误差
- **MSE (Mean Squared Error)**: 均方误差
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **Prediction_Time**: 预测时间

> ⚡ 评测使用 **10个并发线程** 对测试集进行预测

> **位次排序逻辑**：RMSE低 -> 推理时间短 -> 最近提交时间近


### 评分方式

#### 总分：20分

🏆 10分 - Metric得分（基于RMSE性能）  
🏆 10分 - 位次得分（基于排名）

#### 📊 评分规则

采用**线性变换**的方式计算分数：

<table>
<tr>
<th>🌟 等级</th>
<th>📍 标准</th>
<th>💯 得分</th>
</tr>
<tr>
<td align="center"><b>前10%学生</b></td>
<td>leaderboard第10%分位的RMSE和位次</td>
<td align="center"><b>20分</b><br/>(满分)</td>
</tr>
<tr>
<td align="center"><b>中间学生</b></td>
<td>在10%分位线和baseline之间</td>
<td align="center"><b>4-20分</b><br/>(线性插值)</td>
</tr>
<tr>
<td align="center"><b>Baseline</b></td>
<td>baseline的RMSE和位次</td>
<td align="center"><b>4分</b><br/>(2分metric + 2分位次)</td>
</tr>
<tr>
<td align="center"><b>未提交</b></td>
<td>-</td>
<td align="center"><b>0分</b></td>
</tr>
</table>

---

## 📂 项目结构

```
📦 project/
 ┣ 📄 train.csv              # 训练数据
 ┣ 🔧 model.py               # 模型实现
 ┣ 🚀 solution.py            # 推理接口
 ┣ 📋 requirements.txt       # 依赖库
 ┣ 🐧 evaluate-linux         # Linux评测程序
 ┣ 🍎 evaluate-macos         # macOS评测程序
 ┗ 🪟 evaluate-win.exe       # Windows评测程序
```

> 💡 **Baseline**: 使用随机权重，你需要实现自己的训练逻辑来提升性能

---

## 💻 模型实现

### 🎨 1. `model.py` - 模型类

```python
import numpy as np

class Model:
    def __init__(self):
        """初始化模型参数"""
        self.weights = None
        self.bias = None
        
    def predict(self, X):
        """
        Args:
            X: numpy数组, shape (n_samples, n_features)
        Returns:
            numpy数组, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
```

### 🚀 2. `solution.py` - 推理接口

```
class Solution:        
    def forward(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """模型推理接口，接收单条样本数据并返回预测结果
        
        Args:
            sample: 单条样本数据字典，包含ID列及特征列（不含age列）
                示例: {'id': 666336, 'job': 'blue-collar', 'marital': 'married', 
                       'education': 'secondary', 'default': 'no', 'balance': 3595,
                       'housing': 'no', 'loan': 'yes', 'contact': 'unknown', 
                       'day': 3, 'month': 'jul', 'duration': 198, 'campaign': 2,
                       'pdays': -1, 'previous': 0, 'poutcome': 'unknown'}
        
        Returns:
            包含预测结果的字典，格式为: {'prediction': 预测概率值}
        """
        # 1. 特征处理：根据模型要求对样本特征进行转换（如编码、归一化等）
        # 示例：可将字典转换为DataFrame便于处理
        # feature_df = pd.DataFrame([sample])
        
        # 2. 模型加载与推理：使用加载的模型对处理后的特征进行预测
        # 示例：model = Model()  # 假设Model类有加载和预测方法
        # prediction = model.predict(feature_df)
        
        # 3. 结果处理：确保返回值为float类型
        prediction = 0.0  # 此处替换为实际预测逻辑
        
        return {'prediction': float(prediction)}
```
---

## ⚙️ 环境要求

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

**📦 依赖安装**:

```bash
pip install -r requirements.txt
```

---

## 🚀 运行评测

### 📥 1. 下载评测程序
<details>
<summary><b>📖 点击查看详细步骤</b></summary>

1. 🔗 进入GitHub仓库
2. 🏷️ 点击 [release](https://github.com/oneJue/ML-Assignment-1/releases/tag/v16) 标签
3. ⬇️ 下载对应系统的文件：
   - 🐧 **Linux**: [evaluate-linux](https://github.com/oneJue/ML-Assignment-1/releases/download/v16/evaluate-linux)
   - 🍎 **macOS**: [evaluate-macos](https://github.com/oneJue/ML-Assignment-1/releases/download/v16/evaluate-macos)
   - 🪟 **Windows**: [evaluate-win.exe](https://github.com/oneJue/ML-Assignment-1/releases/download/v16/evaluate-win.exe)

</details>

> ⚠️ **重要**：将下载的评测程序放在**项目根目录**（与solution.py、model.py同级）

### ⚙️ 2. 设置环境变量

**🐧 Linux/macOS:**

```bash
export STUDENT_ID='你的学号'
export STUDENT_NAME='你的姓名'
export STUDENT_NICKNAME='你的昵称'
```

💾 持久化：添加到`~/.bashrc`或`~/.zshrc`

**🪟 Windows:**

```cmd
set STUDENT_ID=你的学号
set STUDENT_NAME=你的姓名
set STUDENT_NICKNAME=你的昵称
```

💾 持久化：系统设置 → 环境变量

### ▶️ 3. 运行评测

**🐧 Linux(ubuntu 24.02)**

```bash
chmod +x evaluate-linux
./evaluate-linux
```

**🍎 macOS**

```bash
chmod +x evaluate-macos
./evaluate-macos
```

> ⚠️ macOS 首次运行提示：若系统提示 “无法打开，因为它来自身份不明的开发者”，请按以下步骤操作：
> 点击弹窗中的 “取消”；
> 打开系统设置（System Settings） → 进入隐私与安全性（Privacy & Security）；
> 在页面下方 “安全” 区域找到 “evaluate-macos 已被阻止打开” 的提示，点击右侧 **“仍要打开”**；
> 在确认窗口中再次点击 “打开”，即可正常运行。
> 
**🪟 Windows**

```cmd
evaluate-win.exe
```

---

## 🏆 Leaderboard

<div align="center">

### 🌐 访问地址

**🔗 [http://101.132.193.95:3000](http://101.132.193.95:3000)**

---

### ✨ 功能特性


<div align="center">

📊 实时排名显示

📈 详细指标展示 (MAE/MSE/RMSE/时间)

🔄 未提交同学显示

⏰ 支持多次提交,以截止日期前的最佳成绩为准


</div>


---

### 🎉 祝你取得好成绩！

**📅 记得在11月2日前提交你的最佳成绩！**

---

Made with ❤️ for Machine Learning Education

</div>
