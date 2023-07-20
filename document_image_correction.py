import cv2
import numpy as np


def drowLine(img, lines):
    im = img.copy()
    for i in range(len(lines)):
        l = lines[i][0]
        cv2.line(im, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(im, (l[0], l[1]), 6, (0, 0, 0), 2)
        cv2.circle(im, (l[2], l[3]), 6, (0, 0, 0), 2)
    return im


def display(name, img):
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getABC(x1, y1, x2, y2):
    '''
    x1, y1: 端点1
    x2, y2: 端点2
    return: 端点1和2构成直线的一般方程系数
    '''
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    return A, B, C


def getLinePoint(A1, B1, C1, A2, B2, C2):
    '''
    A1, B1, C1: 直线1
    A2, B2, C2: 直线2
    return: 直线1和2的交点
    '''
    x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
    y = (A1 * C2 - A2 * C1) / (A2 * B1 - A1 * B2)
    return [x, y]


def calc_P2P_Distance(x1, y1, x2, y2):
    '''
    x1, y1: 点1
    x2, y2: 点2
    return: 点1和点2的欧式距离
    '''
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def calc_P2L_Distance(x, y, x1, y1, x2, y2):
    '''
    x, y: 点
    x1, y1, x2, y2: 一条直线的两端端点
    return: 点到该直线的距离
    '''
    A, B, C = getABC(x1, y1, x2, y2)
    denominator = np.sqrt((A ** 2) + (B ** 2))
    if denominator == 0:
        return calc_P2P_Distance(x, y, x1, y1)
    else:
        return abs((A * x) + (B * y) + C) / denominator


def calc_L2L_Distance(xi1, yi1, xi2, yi2, xj1, yj1, xj2, yj2):
    '''
    xi1, yi1, xi2, yi2: 直线1的两端端点
    xi2, yi2, xj2, yj2: 直线2的两端端点
    return: 直线1和直线2的距离
    method: 计算两对端点之间的四种组合的距离，求平均
    '''
    d1 = calc_P2L_Distance(xi1, yi1, xj1, yj1, xj2, yj2)
    d2 = calc_P2L_Distance(xi2, yi2, xj1, yj1, xj2, yj2)
    d3 = calc_P2L_Distance(xj1, yj1, xi1, yi1, xi2, yi2)
    d4 = calc_P2L_Distance(xj2, yj2, xi1, yi1, xi2, yi2)
    return (d1 + d2 + d3 + d4) / 4


def calcK(x1, y1, x2, y2):
    '''
    xi1, yi1, xi2, yi2: 直线的两端端点
    return: 直线的斜率（以角度形式返回）
    '''
    if x2 == x1: return 90
    return np.degrees(np.arctan((y2 - y1) / (x2 - x1)))


def cleanClutter(lines):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    return: 清理杂乱直线后的结果
    method: 1. 扫描所有直线，计算周围与其靠近的直线数目（如果两条直线中某对端点距离小于15，则认为靠近）
            2. 对于一条直线，如果和其靠近的直线数目大于5，则删除该条直线
    '''
    count = np.ones(lines.shape[0])
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            xi1 = lines[i, 0, 0]
            yi1 = lines[i, 0, 1]
            xi2 = lines[i, 0, 2]
            yi2 = lines[i, 0, 3]
            xj1 = lines[j, 0, 0]
            yj1 = lines[j, 0, 1]
            xj2 = lines[j, 0, 2]
            yj2 = lines[j, 0, 3]
            d1 = calc_P2P_Distance(xi1, yi1, xj1, yj1)
            d2 = calc_P2P_Distance(xi2, yi2, xj1, yj1)
            d3 = calc_P2P_Distance(xi1, yi1, xj2, yj2)
            d4 = calc_P2P_Distance(xi2, yi2, xj2, yj2)
            if min(d1, d2, d3, d4) < 15:
                count[i] += 1
                count[j] += 1
    index = count <= 5
    return lines[index]


def merge(line1, line2):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    return: 直线1和直线2合并后的直线
    method: 选择直线1和直线2中距离最远的两个点，连成一条直线返回
    '''
    x11 = line1[0][0]
    y11 = line1[0][1]
    x12 = line1[0][2]
    y12 = line1[0][3]
    x21 = line2[0][0]
    y21 = line2[0][1]
    x22 = line2[0][2]
    y22 = line2[0][3]
    d1 = calc_P2P_Distance(x11, y11, x21, y21)
    d2 = calc_P2P_Distance(x12, y12, x21, y21)
    d3 = calc_P2P_Distance(x11, y11, x22, y22)
    d4 = calc_P2P_Distance(x12, y12, x22, y22)
    d5 = calc_P2P_Distance(x11, y11, x12, y12)
    d6 = calc_P2P_Distance(x21, y21, x22, y22)
    d_max = max(d1, d2, d3, d4, d5, d6)
    if d_max == d1:
        return [x11, y11, x21, y21]
    elif d_max == d2:
        return [x12, y12, x21, y21]
    elif d_max == d3:
        return [x11, y11, x22, y22]
    elif d_max == d4:
        return [x12, y12, x22, y22]
    elif d_max == d5:
        return [x11, y11, x12, y12]
    else:
        return [x21, y21, x22, y22]


def isNear(line1, line2, flag):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    flag: 斜率计算标志（为False时以纵轴为x轴，其余情况以横轴为x轴）
    return: 直线1和直线2在flag指定的模式下是否相近
    method: 1. 判断两直线的斜率是否相近（绝对差小于5）
            2. 计算任意两对端点组成直线的斜率，判断与原直线斜率是否相近（最大绝对差小于5）
            3. 判断两条直线是否相近（距离小于10）
            4. 满足上述条件，返回True；否则，返回False
    '''
    xi1 = line1[0][0]
    yi1 = line1[0][1]
    xi2 = line1[0][2]
    yi2 = line1[0][3]
    xj1 = line2[0][0]
    yj1 = line2[0][1]
    xj2 = line2[0][2]
    yj2 = line2[0][3]
    if not flag:
        xi1, yi1 = yi1, xi1
        xi2, yi2 = yi2, xi2
        xj1, yj1 = yj1, xj1
        xj2, yj2 = yj2, xj2
    ki = calcK(xi1, yi1, xi2, yi2)
    kj = calcK(xj1, yj1, xj2, yj2)
    if abs(ki - kj) < 5:  # 1. 判断两直线倾斜度是否相近
        k1 = calcK(xi1, yi1, xj1, yj1)
        k2 = calcK(xi1, yi1, xj2, yj2)
        k3 = calcK(xi2, yi2, xj1, yj1)
        k4 = calcK(xi2, yi2, xj2, yj2)
        kmax = max(k1, k2, k3, k4)
        if max(abs(kmax - ki), abs(kmax - kj)) < 5:  # 2. 判断四个端点的组合方式中是否相近
            d = calc_L2L_Distance(xi1, yi1, xi2, yi2, xj1, yj1, xj2, yj2)
            if d > 10: return False
            return True
    return False


def isLineNear(line1, line2):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    return: 直线1和直线2是否相近
    method: 两条直线在横轴为x或纵轴为x的任意一种情况下相近，则返回True；否则，返回False
    '''
    if isNear(line1, line2, True) or isNear(line1, line2, False):
        return True
    return False


def mergeLines(lines):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    return: 合并之后的直线集合
    method: 1. 遍历直线集合，遍历中将相近的直线合并起来覆盖原直线
            2. 如果直线集合改变，重新遍历
    '''
    ts = lines.tolist()
    while True:
        isChange = False
        for i in range(len(ts)):
            if ts[i] == -1: continue
            for j in range(i + 1, len(ts)):
                if ts[j] == -1: continue
                if isLineNear(ts[i], ts[j]):
                    ts[i] = [merge(ts[i], ts[j])]
                    ts[j] = -1
                    isChange = True
        if not isChange: break
    ts = [x for x in ts if x != -1]
    return np.array(ts)


def clearOutside(lines, imShape, threshold):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    imShape: 图片尺寸
    threshold: 判断是否为外部直线的阈值
    return: 清理掉外部直线之后的直线集合
    method: 如果一条直线的某个端点与图片边界的距离小于设定阈值，删除该条直线
    '''
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    w = imShape[1]
    h = imShape[0]
    lMax = threshold
    rMax = w - threshold
    tMax = threshold
    bMax = h - threshold
    index = (x1 >= lMax) & (x1 <= rMax) & (y1 >= tMax) & (y1 <= bMax) \
            & (x2 >= lMax) & (x2 <= rMax) & (y2 >= tMax) & (y2 <= bMax)
    return lines[index]


def isKNear(line1, line2):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    return: 直线1和直线2的斜率是否相近
    method: 在以横轴为x轴和以纵轴为x轴的两种模式下，任意一种斜率的绝对差值小于15，则为相近
    '''
    kx1 = calcK(line1[0][0], line1[0][1], line1[0][2], line1[0][3])
    kx2 = calcK(line2[0][0], line2[0][1], line2[0][2], line2[0][3])
    ky1 = calcK(line1[0][1], line1[0][0], line1[0][3], line1[0][2])
    ky2 = calcK(line2[0][1], line2[0][0], line2[0][3], line2[0][2])
    if abs(kx1 - kx2) < 15 or abs(ky1 - ky2) < 15:
        return True
    return False


def clearShort(lines, threshold):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray)
    threshold: 判断是否短线的阈值
    return: 长度大于阈值的直线
    '''
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    index = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > threshold
    return lines[index]


def clearInside(lines):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray)
    return: 清理掉内部直线之后的直线集合
            特殊：每一簇KNear的直线集合，只留两条，其余删除
    method: 1. 遍历直线集合，对每一簇KNear的直线集合，选取两条距离最远的直线，其余删除
            2. 距离长度定义为“calc_L2L_Distance + 0.5 * calc_P2P_Distance(lines1) + 0.5 * calc_P2P_Distance(lines2)”
    '''
    ts = lines.tolist()
    while True:
        isChange = False
        for i in range(len(ts)):
            if ts[i] == -1: continue
            l = i
            r = i
            dmax = 0
            for j in range(i + 1, len(ts)):
                if ts[j] == -1: continue
                if r == l:
                    if isKNear(ts[l], ts[j]):
                        r = j
                        dmax = calc_L2L_Distance(ts[l][0][0], ts[l][0][1], ts[l][0][2], ts[l][0][3],
                                                 ts[r][0][0], ts[r][0][1], ts[r][0][2], ts[r][0][3]) \
                               + 0.5 * calc_P2P_Distance(ts[l][0][0], ts[l][0][1], ts[l][0][2], ts[l][0][3]) \
                               + 0.5 * calc_P2P_Distance(ts[r][0][0], ts[r][0][1], ts[r][0][2], ts[r][0][3])
                    continue
                if not isKNear(ts[l], ts[j]) and not isKNear(ts[r], ts[j]): continue
                dl = calc_L2L_Distance(ts[l][0][0], ts[l][0][1], ts[l][0][2], ts[l][0][3],
                                       ts[j][0][0], ts[j][0][1], ts[j][0][2], ts[j][0][3]) \
                     + 0.5 * calc_P2P_Distance(ts[l][0][0], ts[l][0][1], ts[l][0][2], ts[l][0][3]) \
                     + 0.5 * calc_P2P_Distance(ts[j][0][0], ts[j][0][1], ts[j][0][2], ts[j][0][3])

                dr = calc_L2L_Distance(ts[r][0][0], ts[r][0][1], ts[r][0][2], ts[r][0][3],
                                       ts[j][0][0], ts[j][0][1], ts[j][0][2], ts[j][0][3]) \
                     + 0.5 * calc_P2P_Distance(ts[r][0][0], ts[r][0][1], ts[r][0][2], ts[r][0][3]) \
                     + 0.5 * calc_P2P_Distance(ts[j][0][0], ts[j][0][1], ts[j][0][2], ts[j][0][3])
                if dl > dr:
                    if dl > dmax:
                        dmax = dl
                        ts[r] = -1
                        r = j
                        isChange = True
                        continue
                if dr > dmax:
                    dmax = dr
                    ts[l] = -1
                    l = j
                    isChange = True
                    continue
                ts[j] = -1
                isChange = True
        if not isChange: break
    ts = [x for x in ts if x != -1]
    return np.array(ts)


def choiceBoundingBox(lines):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray)
           特殊：集合中与任意一条直线KNear的直线有且仅有一条
    return: BoundingBox（四条直线，直线0和直线1为一对边，直线2和直线3为一对边）
    method: 选择calc_L2L_Distance最远的两对直线
    '''
    d = []
    ts = lines.tolist()
    for i in range(len(lines)):
        if ts[i] == -1: continue
        for j in range(i + 1, len(lines)):
            if ts[j] == -1: continue
            if isKNear(ts[i], ts[j]):
                d.append(ts[i])
                d.append(ts[j])
                ts[j] = -1
    indexmax1 = 0
    dmax1 = calc_L2L_Distance(d[0][0][0], d[0][0][1], d[0][0][2], d[0][0][3],
                              d[1][0][0], d[1][0][1], d[1][0][2], d[1][0][3])
    indexmax2 = 2
    dmax2 = calc_L2L_Distance(d[2][0][0], d[2][0][1], d[2][0][2], d[2][0][3],
                              d[3][0][0], d[3][0][1], d[3][0][2], d[3][0][3])
    for i in range(4, len(d) - 1, 2):
        dts = calc_L2L_Distance(d[i][0][0], d[i][0][1], d[i][0][2], d[i][0][3],
                                d[i + 1][0][0], d[i + 1][0][1], d[i + 1][0][2], d[i + 1][0][3])
        if dts > dmax1:
            dmax1 = dts
            d[indexmax1] = -1
            d[indexmax1 + 1] = -1
            indexmax1 = i
            continue
        if dts > dmax2:
            dmax2 = dts
            d[indexmax2] = -1
            d[indexmax2 + 1] = -1
            indexmax2 = i
            continue
        d[i] = -1
        d[i + 1] = -1
    d = [x for x in d if x != -1]
    return np.array(d)


def getOrder(Point, Center):
    '''
    Point: 点
    Center: 中心点
    return: 点相对于中心点的位置
    method: 左上角为0，右上角为1，右下角为2，左下角为3
    '''
    # x1 == x2
    if Point[0] == Center[0]:
        # y1 <= y2
        if Point[1] <= Center[1]: return 0
        # y1 > y2
        else: return 2
    # y1 == y2
    if Point[1] == Center[1]:
        # x1 <= x2
        if Point[0] <= Center[0]: return 3
        # x1 > x2
        else: return 1
    # x1 < x2
    if Point[0] < Center[0]:
        # y1 < y2
        if Point[1] < Center[1]: return 0
        # y1 > y2
        else: return 3
    # x1 > x2
    else:
        # y1 < y2
        if Point[1] < Center[1]: return 1
        # y1 > y2
        else: return 2


def getOriginalVertices(lines, flag):
    '''
    lines: 直线集合_BoundingBox（四条直线，直线0和直线1为一对边，直线2和直线3为1一对边）
    flag: 左上角为flag（用于排序）
    return: flag定义下的四个端点（顺时针排好序），格式为np.float32
    '''
    A1, B1, C1 = getABC(lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3])
    A2, B2, C2 = getABC(lines[1][0][0], lines[1][0][1], lines[1][0][2], lines[1][0][3])
    A3, B3, C3 = getABC(lines[2][0][0], lines[2][0][1], lines[2][0][2], lines[2][0][3])
    A4, B4, C4 = getABC(lines[3][0][0], lines[3][0][1], lines[3][0][2], lines[3][0][3])
    Point1 = getLinePoint(A1, B1, C1, A3, B3, C3)
    Point2 = getLinePoint(A3, B3, C3, A2, B2, C2)
    Point3 = getLinePoint(A2, B2, C2, A4, B4, C4)
    Point4 = getLinePoint(A4, B4, C4, A1, B1, C1)
    center = [(Point1[0] + Point2[0] + Point3[0] + Point4[0]) / 4, (Point1[1] + Point2[1] + Point3[1] + Point4[1]) / 4]
    ans = [0, 0, 0, 0]
    ans[getOrder(Point1, center)] = Point1
    ans[getOrder(Point2, center)] = Point2
    ans[getOrder(Point3, center)] = Point3
    ans[getOrder(Point4, center)] = Point4
    flag = flag % 4
    return np.float32(ans[flag:] + ans[0:flag])


def getTargetVertices(w, h, size):
    '''
    w: 目标图像宽度
    h: 目标图像高度
    size: 目标图像尺寸（对角线长度）
    return: 目标图像四个端点（左上角为0，顺时针顺序）
    '''
    k = size / np.sqrt(w ** 2 + h ** 2)
    W = w * k
    H = h * k
    return np.float32([[0, 0], [W, 0], [W, H], [0, H]])


def getWarpedOutput(im0, w, h, flag, size):
    '''
    im0: 输入图像
    w: 目标图像宽度（不确定输入-1）
    h: 目标图像高度（不确定输入-1）
    flag: 左上角为flag（用于排序，不确定输入-1）
    length: 目标图像尺寸（对角线长度）
    return: 矫正后的图像
    '''
    imGray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    # Canny边缘检测 + Hough变换
    edges = cv2.Canny(imGray, 50, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45, minLineLength=30, maxLineGap=15)
    lines = cleanClutter(lines) # 清除噪点
    lines = clearShort(lines, 50) # 清理短线
    lines = mergeLines(lines) # 合并直线
    lines = clearOutside(lines, im0.shape, 10) # 清理外线
    lines = clearShort(lines, 200) # 清理短线
    lines = clearInside(lines) # 清理内线
    lines = choiceBoundingBox(lines) # Bounding Box
    # 获取原图四个端点
    if w == -1 or h == -1 or flag == -1: flag = 0
    Original = getOriginalVertices(lines, flag)
    # 获取目标图像四个端点
    if w == -1 or h == -1 or flag == -1:
        w = (calc_P2P_Distance(Original[0][0], Original[0][1], Original[1][0], Original[1][1])
             + calc_P2P_Distance(Original[2][0], Original[2][1], Original[3][0], Original[3][1])) / 2
        h = (calc_P2P_Distance(Original[0][0], Original[0][1], Original[3][0], Original[3][1])
             + calc_P2P_Distance(Original[1][0], Original[1][1], Original[2][0], Original[2][1])) / 2
    Target = getTargetVertices(w, h, size)
    # 获取单应变换矩阵
    M = cv2.getPerspectiveTransform(Original, Target)
    # 图像矫正
    output = cv2.warpPerspective(im0, M, (int(Target[2, 0]), int(Target[2, 1])))
    return output


def test(i, calc_ratio, size = 800):
    '''
    i: 测试序号
    calc_ratio: 是否自动计算比例
    size: 图像尺寸
    '''
    if i > 7 or i < 0:
        print('image index out of range')
        exit()
    datadir = "./data"
    name = ['Lab3-1.jpg', 'Lab3-2.jpg', 'Lab3-3.jpg', 'Lab3-4.jpg', 'Lab3-5.jpg', 'Lab3-6.jpg', '1.jpg', '2.jpg']
    name = [datadir + '/' + name[i] for i in range(8)]
    w = [17, 11.2, 14, 9.5, 12, 26, -1, -1]
    h = [24, 18.2, 20, 6.5, 8.5, 21, -1, -1]
    flag = [3, 3, 3, 0, 0, 0, -1, -1]
    # 读取图像
    im0 = cv2.imread(name[i], cv2.IMREAD_COLOR)
    if im0 is None:
        print('read image failed')
        exit()
    if calc_ratio:
        display('output_calcRatio', getWarpedOutput(im0, -1, -1, -1, size))
    else:
        display('output_knownRatio', getWarpedOutput(im0, w[i], h[i], flag[i], size))

if __name__ == '__main__':
    for i in range(8):
        test(i, False)
        test(i, True)