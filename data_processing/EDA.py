# coding=utf-8
import pandas as pd
import numpy as np
tr_x = pd.read_csv('/data/q1/data_train.csv')
print tr_x.head()
'''
       用户标识   性别  年龄段  大致消费水平  每月的大致刷卡消费次数 手机品牌    手机终端型号  用户更换手机频次  暴风影音  \
0  21185628  1.0  6.0     7.0        280.0   三星  SM-G9200       2.0     0
1  21309819  0.0  5.0     1.0        110.0   苹果     A1661       3.0     0
2  20575927  0.0  5.0     1.0         70.0   苹果     A1661       1.0     0
3  20770740  1.0  7.0     3.0        496.0   苹果     A1586       1.0     0
4  20316679  2.0  8.0     2.0         86.0   小米   2015021       0.0     0

   乐视网    ...      访问论坛网站的次数  访问问答网站的次数  访问阅读网站的次数  访问新闻网站的次数  访问教育网站的次数  \
0    0    ...              0          0          0          0          0
1    0    ...              0          0          0          0          0
2    0    ...              0          1          0          0          0
3    0    ...              0          0          0          0          0
4    0    ...              0          0          0          0          0

   访问孕期网站的次数  访问育儿网站的次数  访问金融网站的次数  访问股票网站的次数  访问游戏网站的次数
0          0          0          3          0          0
1          0          0          0          0          0
2          0          0          0          0          0
3          0          0          6          0          0
4          0          0          0          0          0
'''
print('train set length:{0}'.format( len(tr_x) ))
#train set length:844984

a = tr_x.isnull().any()
df = pd.DataFrame()
df['col'] = list(a.index)
df['isnull'] = list(a.values)
print df[df['isnull']==True]
'''
        col isnull
5      手机品牌   True
7  用户更换手机频次   True
'''
tr_x['用户更换手机频次'] = tr_x['用户更换手机频次'].fillna(0)
# a =  tr_x.groupby('手机品牌')['用户更换手机频次'].count().reset_index()
# print a.sort_values('用户更换手机频次',ascending=False).head(10)
'''
    手机品牌  频次
814   苹果    361441
431   华为     92666
561   小米     71140
324   三星     46905
676   欧珀     44564
816   荣耀     38518
782   维沃     38317
932   魅族     19488
43   CIB     12603
358   乐视      9818
'''
tr_x['手机品牌'] = tr_x['手机品牌'].fillna('未知')
print len(tr_x[tr_x['手机品牌']=='未知']) #612
def do_brand( x ):
    if x in ['苹果','华为','小米','三星','欧珀','荣耀','维沃','魅族','CIB','乐视']:
        return x
    return '其他'
mb = pd.get_dummies( tr_x['手机品牌'].apply( do_brand ),prefix='mobile_brand' )
tr_x = pd.concat( [tr_x,mb],axis=1,join='inner' )
del tr_x['手机品牌']

y = pd.read_csv('/data/q1/label.csv')['是否去过迪士尼'].values
print y.head()
'''
    用户标识  是否去过迪士尼
0  21185628        0
1  21309819        0
2  20575927        1
3  20770740        0
4  20316679        0
'''
print( 'len(class==1)/len(class==0)={0}/{1}'.format( len(y[y['是否去过迪士尼']==1]) ,len(y[y['是否去过迪士尼']==0]) ) )
#len(class==1)/len(class==0)=81164/763820


#独热编码：性别，手机品牌，手机终端型号，漫入，漫出
print tr_x[['性别','手机品牌']].groupby('性别').count()
sex = pd.get_dummies( tr_x['性别'],prefix='sex' )
tr_x = pd.concat([tr_x,sex],axis=1,join='inner') #axis=1 是行
del tr_x['性别']
del tr_x['手机终端型号']
del tr_x['漫入']
del tr_x['漫出']

'''
       手机品牌
性别
0.0  347223
1.0  200291
2.0  297470
'''
print tr_x[['手机终端型号','手机品牌']].groupby('手机终端型号').count().reset_index().sort_values('手机品牌',ascending=False).head(10)
'''
                     手机终端型号   手机品牌
568                   A1586  61995
574                   A1661  51643
559                   A1524  44901
579                   A1700  39474
573                   A1660  35682
578                   A1699  26797
563                   A1530  15967
5781      MEXIKO-NAD6200CHN  11868
4147  IPHONE6S(A1688)(疑似水货)  10520
5964                  MU509   9250
'''
print tr_x[['手机终端型号','手机品牌']].groupby('手机终端型号').count().reset_index().sort_values('手机品牌',ascending=False).tail(10)
'''
               手机终端型号  手机品牌
4813     LENOVO A630E     1
4762   LENOVO A3300-T     1
4805     LENOVO A5890     1
4800      LENOVO A580     1
4797  LENOVO A5500-HV     1
4795      LENOVO A529     1
4788     LENOVO A395E     1
4769      LENOVO A360     1
4763     LENOVO A330E     1
9935          雅器 S900     1
'''
tr_x[['漫出省份','手机品牌']].groupby('漫出省份').count().sort_values('手机品牌',ascending=False).reset_index().head(20)
'''
       漫出省份    手机品牌
0         无  475722
1       江苏省  104985
2       浙江省   55420
3   江苏省,浙江省   12839
4   浙江省,江苏省   12732
5       安徽省   10594
6   安徽省,江苏省    7857
7   江苏省,安徽省    7826
8       广东省    6808
9       北京市    4958
10      河南省    3855
11      山东省    3652
12      江西省    2695
13      福建省    2573
14      湖北省    2529
15      四川省    2430
16      湖南省    1814
17  浙江省,安徽省    1661
18  安徽省,浙江省    1618
19      陕西省    1614
江苏省，浙江省，安徽省，广东省，北京市，河南省，山东省，江西省，福建省，湖北省，无
'''
province = {'江苏省':0, '浙江省':1, '安徽省':2, '广东省':3, '北京市':4, '河南省':5, '山东省':6, '江西省':7, '福建省':8, '湖北省':9}
def fn( x ):
    for i in x.split(','):
        if i in province : return province[i]
        else: return -1
tp = pd.get_dummies( tr_x['漫出省份'].apply( fn ),prefix='to_province' )
tr_x['num_province']  = tr_x['漫出省份'].apply( lambda x:len(str(x).split(','))  )

tr_x[['漫出省份','手机品牌']].groupby('漫出省份').count().sort_values('手机品牌',ascending=False).reset_index().tail(10)
'''
                      漫出省份  手机品牌
24170          广东省,广西省,福建省     1
24171      广东省,广西省,江苏省,浙江省     1
24172      广东省,广西省,湖北省,江苏省     1
24173      广东省,广西省,浙江省,福建省     1
24174  广东省,广西省,河南省,浙江省,陕西省     1
24175  广东省,广西省,江西省,浙江省,海南省     1
24176      广东省,广西省,江西省,安徽省     1
24177      广东省,广西省,江苏省,贵州省     1
24178  广东省,广西省,江苏省,浙江省,河北省     1
24179  黑龙江,陕西省,青海省,江苏省,浙江省     1
'''

# ###############################################################################################################
# 各网站总pv
# 关注网站类型数
# 300个app中非空列数
netcate_list = ['访问视频网站的次数', '访问音乐网站的次数', '访问图片网站的次数'
    , '访问体育网站的次数', '访问健康网站的次数', '访问动漫网站的次数'
    , '访问搜索网站的次数', '访问生活网站的次数', '访问购物网站的次数'
    , '访问房产网站的次数', '访问地图网站的次数', '访问餐饮网站的次数'
    , '访问汽车网站的次数', '访问旅游网站的次数', '访问综合网站的次数'
    , '访问IT网站的次数', '访问聊天网站的次数', '访问交友网站的次数'
    , '访问社交网站的次数', '访问通话网站的次数', '访问论坛网站的次数'
    , '访问问答网站的次数', '访问阅读网站的次数', '访问新闻网站的次数'
    , '访问教育网站的次数', '访问孕期网站的次数', '访问育儿网站的次数'
    , '访问金融网站的次数', '访问股票网站的次数', '访问游戏网站的次数']

tr_x['sum_pv'] = 0
tr_x['net_cnt'] = 0
tr_x['app_cnt'] = 0
for f in netcate_list:
    tr_x['sum_pv'] += tr_x[f]
    tr_x['net_cnt'] += tr_x[f].apply(lambda x: x != 0)
for a in range(8, 308):
    tr_x['app_cnt'] += tr_x.ix[:, a].apply(lambda x: x != 0)


##题2 数据
xls2 = pd.ExcelFile("/data/q2/数据集7_上海主要景点名单及POI经纬度信息.xlsx")
xls2_sheet1 = xls2.parse("Sheet1")
print len(xls2_sheet1) #81
xls2_sheet1['t'] = xls2_sheet1[u'经度']
xls2_sheet1[u'经度'] = xls2_sheet1[u'纬度']
xls2_sheet1[u'纬度'] = xls2_sheet1['t']
del xls2_sheet1['t']
xls2_sheet1.to_csv('/data/q2/scenic_spot.csv',encoding='utf8',index=False)
print xls2_sheet1.head()
'''
        景点名称    级别        经度        纬度
0  上海国际旅游度假区  AAAA  31.16112  121.6840
1   上海共青森林公园  AAAA  31.32654  121.5580
2    上海召稼楼景区   AAA  31.08094  121.5556
3      朱家角古镇  AAAA  31.11547  121.0632
4    大宁郁金香公园    AA  36.43264  110.7108
'''
#数据集6_上海主要景点瞬时最大承载量.xls
xls6 = pd.ExcelFile("/data/q2/数据集6_上海主要景点瞬时最大承载量.xls")
xls6_sheet1 = xls6.parse(xls6.sheet_names[0])
print xls6_sheet1.head()
'''
        景点名称  瞬时最大承载量
0     碧海金沙景区    18000
1    大宁郁金香公园    18325
2     东方假日田园     2800
3  东方明珠广播电视塔    15000
4   东平国家森林公园    53817
'''
xls6_sheet1.to_csv('/data/q2/scenic_spot_max_loads.csv',index=False,encoding='utf8')

xls5 = pd.ExcelFile("/data/q2/数据集5_上海主要景点4-5月游客数量情况.xlsx")
xls5_sheet1 = xls5.parse(xls5.sheet_names[0])
print xls5_sheet1.head()
'''
    景点名称    人数                  时间 舒适度等级
0  朱家角古镇   570 2017-04-01 08:50:00  非常舒适
1  朱家角古镇   813 2017-04-01 09:20:00  非常舒适
2  朱家角古镇  1192 2017-04-01 09:50:00  非常舒适
3  朱家角古镇  1508 2017-04-01 10:20:00  非常舒适
4  朱家角古镇  1766 2017-04-01 10:50:00  非常舒适
'''
print xls5_sheet1.shape #(71850, 4)
crowd_event =  xls5_sheet1[xls5_sheet1[u'舒适度等级']==u'非常拥挤']
crowd_event['datetime'] = pd.to_datetime(crowd_event[u'时间'])
crowd_event['weekday'] = crowd_event['datetime'].apply(lambda x:1+x.weekday())#Monday == 0 ... Sunday == 6
crowd_event['hour'] = crowd_event['datetime'].apply(lambda x:x.hour)
crowd_event_cnt = crowd_event.groupby( [u'景点名称','weekday','hour'] )[u'舒适度等级'].count().reset_index()
crowd_event_cnt.columns = [u'景点名称','weekday','hour','count']
crowd_event_cnt = crowd_event_cnt.sort_values([u'景点名称','hour'])


def fn(x):
    a = list(set(x))
    a.sort()
    a = map(str, a)
    return '/'.join(a)
a = crowd_event_cnt.groupby(u'景点名称')['hour'].apply( lambda x:fn(x) ).reset_index()
b = crowd_event_cnt.groupby(u'景点名称')['weekday'].apply( lambda x:fn(x) ).reset_index()
#################################各景点拥堵事件统计===============================
crowd_ = pd.merge( a,b,on=u'景点名称' )
crowd_['crowd_dur'] = crowd_['hour'].apply(lambda s:len(str(s).split('/')) )
print crowd_
'''
          景点名称                              拥堵时段        拥护日期        拥堵时长
0     上海中医药博物馆                             10/14            3/4          2
1     上海共青森林公园                                14              6          1
2        上海动物园                 10/11/12/13/14/15            1/7          6
3    上海国际旅游度假区           11/12/13/14/15/16/17/18      1/4/5/6/7          8
4        上海植物园                          14/15/16          1/2/7          3
5      上海海洋水族馆                          13/14/15              7          3
6      上海炮台湾景区                                14              1          1
7      上海玻璃博物馆                       13/14/15/16      2/3/4/5/6          4
8      上海田子坊景区  10/11/12/13/14/15/16/17/18/19/20  1/2/3/4/5/6/7         11
9        上海科技馆                 10/11/12/13/14/15          1/6/7          6
10        上海豫园                       13/14/15/16          4/5/6          4
11    上海都市菜园景区            9/10/11/12/13/14/15/16        1/5/6/7          8
12    上海闵行体育公园                             10/15              1          2
13       上海闻道园                             11/13            1/7          2
14   上海马陆葡萄艺术村                          13/14/15              7          3
15     大宁郁金香公园                 11/12/13/14/15/16            1/7          6
16        新场古镇                          12/13/14            1/2          3
17       朱家角古镇                          12/13/14            1/7          3
18  金茂大厦88层观光厅                    16/17/18/20/21          1/6/7          5
19    长风海洋世界景区                       13/14/15/16          5/6/7          4
'''

user_loc = pd.read_csv("/data/q2/数据集1_用户地理位置.csv")
print user_loc.head()
'''
         日期  时段      用户标识         经度      纬度
0  20170401   0  21714645  121.48258  31.239
1  20170401   0  20734366  121.48258  31.239
2  20170401   0  20000169  121.48258  31.239
3  20170401   0  20368284  121.48258  31.239
4  20170401   0  21224649  121.48258  31.239
'''
user_loc.to_csv('/data/q2/user_loc_log.csv')
#xls = pd.ExcelFile("/data/q2/数据集8_上海主要酒店名单及POI经纬度信息.xlsx")
# sheet1 = xls.parse("Sheet1")
# print sheet1.head()
# '''
#                 酒店名称         经度        纬度
# 0  碧云花园服务公寓 - 碧云国际社区  121.58078  31.24325
# 1        上海新金桥智选假日酒店  121.61271  31.27571
# 2           外滩帕奇艺术酒店  121.48567  31.23424
# 3            上海外滩悦榕庄  121.50235  31.25358
# 4             上海新城酒店  121.48280  31.23705
# '''

# xls2_sheet2 = xls2.parse("Sheet2")
# print xls2_sheet2.head()
# '''
#  1 上海国际旅游度假区  31.16112   121.684       上海市浦东新区探索路  上海市 上海市.1 浦东新区    310115
# 0  2  上海共青森林公园  31.32654  121.5580           上海市杨浦区  上海市   上海市  杨浦区  310110.0
# 1  3   上海召稼楼景区  31.08094  121.5556  上海市闵行区沈杜公路2071号  上海市   上海市  闵行区  310112.0
# 2  4     朱家角古镇  31.11547  121.0632    上海市青浦区新风路168号  上海市   上海市  青浦区  310118.0
# 3  5   大宁郁金香公园  36.43264  110.7108        山西省临汾市大宁县  山西省   临汾市  大宁县  141030.0
# 4  6     上海植物园  31.15250  121.4502           上海市徐汇区  上海市   上海市  徐汇区  310104.0
# '''

# xls4 = pd.ExcelFile("/data/q2/数据集4_旅游团队成员信息-到上海旅游的旅行团.xlsx")
# print len(xls4.sheet_names) #3
# xls4_sheet1 = xls4.parse(xls4.sheet_names[0])
# print xls4_sheet1.head()
# '''
#     团队编号          游客ID  性别  年龄段
# 0  team1  tourist20313   0  6.0
# 1  team1  tourist20314   1  5.0
# 2  team1  tourist20315   1  6.0
# 3  team1  tourist20316   0  8.0
# 4  team1  tourist20317   1  8.0
# '''
# xls4_sheet2 = xls4.parse(xls4.sheet_names[2])
# print xls4_sheet2.head()
#
# xls4_ = pd.ExcelFile("/data/q2/数据集4_旅游团队成员信息-从上海出发的旅行团.xlsx")
# print len(xls4_.sheet_names)
# xls4__sheet1 = xls4_.parse(xls4_.sheet_names[0])
# print xls4__sheet1.head()
# '''
#     团队编号           游客ID  性别  年龄段
# 0  team1  tourist175276   1  6.0
# 1  team1  tourist175277   1  5.0
# 2  team1  tourist175278   1  4.0
# 3  team1  tourist175279   0  1.0
# 4  team1  tourist175280   1  7.0
# '''
# xls4__sheet2 = xls4_.parse(xls4_.sheet_names[1])
# print xls4__sheet2.head()
#
# xls3 = pd.ExcelFile("/data/q2/数据集3_旅游团队信息-到上海旅游的旅行团.xlsx")
# print len(xls3.sheet_names)
# xls3_sheet1 = xls3.parse(xls3.sheet_names[0])
# print xls3_sheet1.head()
# '''
#     团队编号              线路名称  出行天数       出发日期       返回日期  团队人数 客源地  行程天数   行程  \
# 0  team1  北京 上海 西安  14日休闲游    14 2016-04-22 2016-05-05     7  伊朗     1  北京市
# 1  team1  北京 上海 西安  14日休闲游    14 2016-04-22 2016-05-05     7  伊朗     2  上海市
# 2  team1  北京 上海 西安  14日休闲游    14 2016-04-22 2016-05-05     7  伊朗     3  上海市
# 3  team1  北京 上海 西安  14日休闲游    14 2016-04-22 2016-05-05     7  伊朗     4  上海市
# 4  team1  北京 上海 西安  14日休闲游    14 2016-04-22 2016-05-05     7  伊朗     5  上海市
#
#   出发城市 到达城市 Unnamed: 11
# 0   伊朗   北京         NaN
# 1   北京   北京         NaN
# 2   北京   北京         NaN
# 3   北京   北京         NaN
# 4   北京   上海         NaN
# '''
# user_tag = pd.read_csv("/data/q2/数据集2_用户标签.csv")
# print user_tag.head()

# hot_spot = user_loc.groupby(['经度','纬度'])['时段'].count().reset_index().sort_values('时段',ascending=False).head(10)
# '''
#              经度         纬度      时段
# 33   121.469350  31.223280  557805
# 70   121.473500  31.237630  467784
# 55   121.471800  31.218325  407136
# 106  121.488350  31.227660  400053
# 38   121.470080  31.218150  361490
# 24   121.468713  31.235458  347105
# 9    121.465830  31.224160  335276
# 37   121.470075  31.235014  315383
# 113  121.655429  31.145725  309968
# 7    121.465800  31.220700  293717
# '''
# points = user_loc[['经度','纬度']].drop_duplicates()
# points.to_csv('/data/cache/points.csv',index=False)
# key = '945ae54516979776ab3ee717012c24d4' #请使用你申请的公钥
# regeo_url = 'http://restapi.amap.com/v3/geocode/regeo?key={key}&location={lonlat}&extensions=all&batch=true'
# points = pd.read_csv('../q2/points.csv')
# address_list = []
# def getHtml(url):
#     '''
#     读取查询地点对应的经纬度信息
#     :param url: 查询地点的request
#     :return: [lon,lat]
#     '''
#     page = urllib.urlopen(url)                #访问网页
#     data = page.readline()                         #读入数据
#     data_dic = json.loads(data)                    #转换成python->字典
#     data_dic_regeocodes = data_dic['regeocodes'][0]   #获取geocodes信息，也是以字典存储
#     data__dic_location = data_dic_regeocodes['formatted_address']  # 获取location信息
#     location = data__dic_location  #处理locaiton成为List
#     return location                                 #返回信息
# for i in range( len(points) ):
#     lonlat = str(points.ix[i,:]['经度']) +','+ str(points.ix[i,:]['纬度'])
#     url = regeo_url.format(key=key, lonlat=lonlat)
#     print url
#     address_list.append( getHtml( url ) )
# points['地址'] = address_list
# def fn( x ):
#     for i in ['上海迪士尼乐园','外滩','豫园城隍庙','人民广场','新天地']:
#         if i in x:
#             return i
#     return '其他'
# points['景点'] = points['地址'].apply( fn )
# points.to_csv('/data/cache/point.csv',index=False,encoding='utf-8')

