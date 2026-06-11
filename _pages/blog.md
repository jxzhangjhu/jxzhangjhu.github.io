---
layout: default
permalink: /blog/
title: Blog
nav: false
nav_order: 3
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 10
  sort_field: date
  sort_reverse: true
  trail:
    before: 1
    after: 3
---

<div class="post blog-lillog">

  <div class="blog-intro">
    <h1>👋 Welcome to Jiaxin's Blog</h1>
    <p>
      I document my notes and writings on AI research, LLMs, and engineering here.
      A mix of long-form posts hosted on this site and selected external articles.
    </p>
  </div>

  <ul class="post-list">

    {%- if page.pagination.enabled -%}
      {%- assign postlist = paginator.posts -%}
    {%- else -%}
      {%- assign postlist = site.posts -%}
    {%- endif -%}

    {% for post in postlist %}

    {% if post.read_time %}
      {% assign read_time = post.read_time %}
    {% elsif post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <li>
      <h2 class="lillog-title">
        {% if post.redirect == blank %}
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% elsif post.redirect contains '://' %}
          <a class="post-title" href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }}</a>
          <i class="fas fa-external-link-alt fa-xs lillog-ext"></i>
        {% else %}
          <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
        {% endif %}
      </h2>

      {% if post.description %}<p class="lillog-summary">{{ post.description }}</p>{% endif %}

      <p class="post-meta lillog-meta">
        {{ post.date | date: '%B %-d, %Y' }}
        &nbsp; &middot; &nbsp; {{ read_time }} min read
        {%- if post.external_source %} &nbsp; &middot; &nbsp; {{ post.external_source }}{%- endif %}
        {%- if tags != "" %} &nbsp; &middot; &nbsp;
          {% for tag in post.tags %}<a href="{{ tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}"><i class="fas fa-hashtag fa-sm"></i> {{ tag }}</a>&nbsp;{% endfor %}
        {%- endif %}
      </p>
    </li>

    {% endfor %}
  </ul>

  {%- if page.pagination.enabled -%}
    {%- include pagination.html -%}
  {%- endif -%}

</div>
