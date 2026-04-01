// Настройки документа
#set document(
  title: "Курсовая работа",
  author: "Автор",
)

// Настройки страницы
#set page(
  paper: "a4",
  margin: (
    left: 2.5cm,
    right: 1.0cm,
    top: 2.0cm,
    bottom: 2.0cm,
  ),
  numbering: "1",
)

// Настройки текста
#set text(
  font: "New Computer Modern",
  size: 12pt,
  lang: "ru",
)

// Междустрочный интервал 1.5
#set par(
  leading: 0.65em,
  first-line-indent: 1.25cm,
  justify: true,
)

// Настройки заголовков
#set heading(numbering: "1.1")

// Главы первого уровня жирным шрифтом
#show heading.where(level: 1): it => {
  set text(weight: "bold")
  it
}

// Настройки для рисунков и таблиц с нумерацией по секциям
#show figure.where(kind: image): set figure(numbering: num => {
  let h = counter(heading).get().first()
  numbering("1.1", h, num)
})

#show figure.where(kind: table): set figure(numbering: num => {
  let h = counter(heading).get().first()
  numbering("1.1", h, num)
})

// Настройки для уравнений с нумерацией по секциям
#set math.equation(numbering: num => {
  let h = counter(heading).get().first()
  numbering("(1.1)", h, num)
})

// Настройки для списков (1. 1.1. 1.1.1.)
#set enum(indent: 1em, numbering: "1.1.1.")

// Настройки для ссылок (синий цвет)
#show link: set text(fill: blue)
#show cite: set text(fill: blue)

// Титульный лист
#include "title_kr.typ"
#pagebreak()

// Начинаем нумерацию со страницы 2
#counter(page).update(2)

// Содержание (оглавление с черными ссылками)
#show outline.entry.where(level: 1): it => {
  set text(fill: black, weight: "bold")
  it
  v(0.65em)
}

#show outline.entry: it => {
  set text(fill: black)
  it
}

#outline(
  title: [Содержание],
  indent: auto,
)

#pagebreak()

// Аннотация
#heading(numbering: none, outlined: true)[Аннотация]

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

// Ключевые слова
#heading(numbering: none, outlined: true)[Ключевые слова]

Глубинное обучение, разреживание моделей, рекуррентные нейронные сети

#pagebreak()

// ========== ОСНОВНАЯ ЧАСТЬ ==========

= Примеры

== Ссылки на статьи

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ссылки на статьи оформляются с помощью встроенной библиографии Typst, например @example_conference. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua @example_journal. 

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris @example_book. Duis aute irure dolor in reprehenderit @example_arxiv. При работе с онлайн ресурсами не забывайте указывать дату обращения @example_website#footnote[Дополнительная информация доступна по ссылке: #link("https://example.com"), дата обр. 30.01.2026].

== Рисунки

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Все рисунки в тексте должны иметь подписи и вы на них должны ссылаться в тексте. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

#figure(
  rect(width: 80%, height: 8cm, fill: rgb("#f0f0f0"), stroke: 1pt)[
    #align(center + horizon)[
      #text(size: 14pt, fill: gray)[График или изображение]
    ]
  ],
  caption: [Пример графика. Тут должна быть подпись, поясняющая что происходит на рисунке.],
) <fig:example>

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. На рисунке @fig:example показан пример. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

== Таблицы

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Все таблицы в тексте тоже должны иметь подписи и вы на них должны ссылаться в тексте. Например, в таблице @table:example показаны результаты примерного эксперимента.

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    stroke: none,
    table.hline(stroke: 1pt),
    table.cell(rowspan: 2)[], 
    table.cell(colspan: 3)[*Val*],
    table.cell(colspan: 3)[*Test*],
    
    table.hline(start: 1, end: 4, stroke: 0.5pt),
    table.hline(start: 4, end: 7, stroke: 0.5pt),
    
    [*Prec*], [*Rec*], [*F1*], [*Prec*], [*Rec*], [*F1*],
    
    table.hline(stroke: 0.5pt),
    
    [запуск 1], [0.4894], [0.3775], [0.4263], [0.4824], [0.3683], [0.4177],
    [запуск 2], [0.4887], [0.3739], [0.4237], [0.4891], [0.3724], [0.4228],
    [запуск 3], [0.4820], [0.3751], [0.4219], [0.4838], [0.3677], [0.4178],
    
    table.hline(stroke: 0.5pt),
    
    [*среднее*], [*0.4867*], [*0.3755*], [*0.4239*], [*0.4851*], [*0.3695*], [*0.4195*],
    [*дисперсия*], [0.0041], [0.0019], [0.0022], [0.0036], [0.0025], [0.0029],
    
    table.hline(stroke: 1pt),
  ),
  caption: [Пример таблицы. Тут должна быть подпись, поясняющая что происходит в таблице.],
) <table:example>

Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

== Формулы

Формулы стоит центрировать, а также нумеровать, если вы ссылаетесь на них в тексте. Также не забывайте пояснять все обозначения в формулах. Например, запишем следующую задачу оптимизации:

$ theta^* = min_(theta) F(theta), $ <eq:optimization>

где $F$ -- квадратичная функция от параметра $theta$. При необходимости, далее в тексте можно сослаться на формулу @eq:optimization. Lorem ipsum dolor sit amet, consectetur adipiscing elit.

= Основная часть работы

== Постановка задачи

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.

== Методология

Totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit.

Sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.

=== Подход 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua:

+ Первый элемент списка
+ Второй элемент списка
+ Третий элемент списка

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

=== Подход 2

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

== Результаты

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

= Заключение

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident.

#pagebreak()

// ========== БИБЛИОГРАФИЯ ==========
#bibliography("refs.bib", title: "Список литературы (или источников)", style: "ieee")

#pagebreak()

// ========== ПРИЛОЖЕНИЕ ==========
#set heading(numbering: "A.1", supplement: [Приложение])
#counter(heading).update(0)

= Пример секции аппендикса

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

