---
display_categories: [generative models, machine autonomy, interpretable machine learning, scene understanding]
horizontal: false
importance: 3
---
<div>
This page lists some representative research works that my students and I do. We are developing machine learning methods for computer vision and machine autonomy. Particularly I am interested in exploring research topics to facilitate human-AI interactions. Some examples are the <a href="https://genforce.github.io/sefa/">human-in-the-loop image editing</a> and the <a href="https://decisionforce.github.io/HACO">human-in-the-loop machine learning</a>. I am also interested in understanding human-centric properties of AI models beyond their performance, such as <a href="http://cnnlocalization.csail.mit.edu/">explainability</a>, <a href="http://netdissect.csail.mit.edu/">interpretability</a>, <a href="https://genforce.github.io/higan/">steerability</a>.

<br>
<br>Please also check out the dedicated research portfolio pages: <a href="https://genforce.github.io/">GenForce</a> on generative modeling, <a href="https://metadriverse.github.io/">MetaDriverse</a> on machine autonomy with application to autonomous driving, intelligent transportation, and human-AI shared control. 

</div>

<div class="projects">
  {% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
    {% for category in page.display_categories %}
      <h2 class="category">{{category}}</h2>
      {% assign categorized_projects = site.projects | where: "category", category %}
      {% assign sorted_projects = categorized_projects | sort: "importance" %}
      <!-- Generate cards for each project -->
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-2">
          {% for project in sorted_projects %}
            {% include projects_horizontal.html %}
          {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="grid">
          {% for project in sorted_projects %}
            {% include projects.html %}
          {% endfor %}
        </div>
      {% endif %}
    {% endfor %}

  {% else %}
  <!-- Display projects without categories -->
    {% assign sorted_projects = site.projects | sort: "importance" %}
    <!-- Generate cards for each project -->
    {% if page.horizontal %}
      <div class="container">
        <div class="row row-cols-2">
        {% for project in sorted_projects %}
          {% include projects_horizontal.html %}
        {% endfor %}
        </div>
      </div>
    {% else %}
      <div class="grid">
        {% for project in sorted_projects %}
          {% include projects.html %}
        {% endfor %}
      </div>
    {% endif %}

  {% endif %}

</div>
