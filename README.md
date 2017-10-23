# GMDH
Group Method of Data Handling

# Описание возможностей

##  Функции


gmdh.**learn(** *x, y, c, F=6, ref_f='lin', crit_f='ssq', max_lvl=3, regularization=False, prnt=True* **)**

Оценивание коэффициентов опорных функций методом наименьших квадратов и селекция *F* наилучших на каждом 
ряду селекции, число которых задаётся параметром *max_lvl*.

**Параметры:** 

x: array-like

    Матрица входных данных, в которой каждый столбец является переменной, а строка -- измерением.   
                
y: array-like

    1-D вектор выходной величины.
                
c: array-like

    Логический вектор, состоящий из 0 и 1. 1 соответствует точкам обучающей последовательности, а
    0 -- точкам проверочной последовательности.
                
F: int, optional

    Свобода выбора - число наилучших отбираемых частных описаний.
                
ref_f: {'lin', 'mul', 'squ'}, optional

    Определяет вид опорной функции:
    
    'lin' -- линейная опорная функция; 
    'mul' -- мультилинейная;
    'squ' -- квадратичная.
                
crit_f: {'ssq', 'rss', 'nrss', 'mimax', 'prc'}, optional

    Задаёт критерий качества, вычисляемый для каждого частного описания по проверочной 
    последовательности. Используется в качестве критерия селекции.
    ssq -- сумма квадратов отклонений;
    rss -- среднеквадратичная ошибка;
    nrss -- нормированная среднеквадратичная ошибка;
    mimax -- максимум абсолютной ошибки;
    prc -- относительная ошибка (проценты).
                
max_lvl: int, optional

    Максимальное количествов рядов селекции.
                
regularization: bool, optional

    Вычисление коэффициентов частных описаний с использованием регуляризации.
                
prnt: bool, optional

    Включение/отключение вывода информации о работе алгоритма: номер ряда селекции, минимальная и 
    средняя ошибки на проверочной последовательности, коэффициенты частных описаний.

**Выходные данные:**

y: array-like

    Матрица результата обучения: номера использованных признаков, коэффициенты частных описаний и их
    ошибки на проверочной последовательности.
    
---    

gmdh.**predict_reg(** *res_matrix, point0, ref_f='lin', lvl=1, num=0* **)**

Вычисление точек предсказания на основе результата работы функции *learn* при параметре regularization=True.

**Параметры:**

res_matrix: array-like

    Результат работы функции learn.
                
point0: array-like

    Матрица входных данных, задающая точки, для которых необходимо провести предсказание.
                
ref_f: {'lin', 'mul', 'squ'}, optional

    Необходимо указать вид опорной функции, использованной при вычислении res_matrix.
                
lvl: int, optional

    Номер ряда селекции.
                
num: int, optional

    Номер частного описания.
                
**Выходные данные:**

y: array-like

    Вектор точек предсказания.
                
---

gmdh.**predict(** *res_matrix, point0, ref_f='lin', lvl=1, num=0* **)**

Вычисление точек предсказания на основе результата работы функции *learn* **без** регуляризации (*regularization=False*). Параметры этой функции не отличаются от параметров функции ***gmdh.predict_reg***.

---

gmdh.**gabor(** *x, m* **)**

Генерирование полинома Колмогорова-Габора из матрицы данных $X$ (за исключением свободного члена).

**Параметры:**

x: array-like

    2-D матрица входных данных.
                
m: int

    Стeпень полинома.
                    
**Выходные данные:**

y: array-like

    2-D матрица, каждый столбец которой отражает член полинома.        

---
