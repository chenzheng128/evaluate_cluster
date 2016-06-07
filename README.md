主程序是 evaluate_class.py

该程序中首先定义了一个 evaluate类 ，此类包含了所有的结果分析方法

类中需要外部调用的函数有三个：
	EvaCF(thre, n)
	EvaClu(user_genres_data, movie_genres_data)
	Q()

其中：
	EvaCF 用来进行 协同预测 评价
	EvaClu 在 社区中用户观看电影种类方面 进行评价
	Q 用来进行 模块度 评价


数据文件存放在 ml-1m 文件夹中


pyfile文件夹中存放一些测试以及数据处理文件

其中：
	collaborative_filtering.py 是利用 协同预测 做矩阵填充的文件，生成predictedratings.dat
	cut_data.py 用来切割数据集
	evaluate_cf.py 是早期用来测试 协同过滤 的文件
	MovieGenres.py 文件是早期用来测试 电影-电影种类 的文件



总之，只要有evaluate_class.py 以及  ml-1m文件夹 中的数据，程序就可以运行了。



PS:
	evaluate.py是未经过类封装的、早期用来做评价的文件
	mylpa.py是根据lpa算法思想我自己写的使用lpa进行社区发现的文件。社区发现效果十分差。
	test.py是测试用文件。一些小功能一般会写一个test.py来做。