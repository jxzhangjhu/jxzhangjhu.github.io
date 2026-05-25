---
layout: aboutnew
title: About
permalink: /
subtitle: >
  <p>Research Lead<br>Salesforce AI Research</p>

profile:
  align: right
  image: prof_pic.jpg
  image_circular: true # crops the image to make it circular
  address: 

news: true
recent_talks: true   # latest 5 talks (data: _data/talks.yml)
conference_travel: true  # conference travel list
awards: true
professional_services: true  # AC, reviewers, etc.
selected_papers: true
social: false  # includes social icons at the bottom of the page
importance: 1
nav: false  # About is hardcoded in header.html, so set to false to avoid duplication
---

<!-- Write your biography here. Tell the world about yourself. Link to your favorite [subreddit](http://reddit.com). You can put a picture in, too. The code is already in, just name your picture `prof_pic.jpg` and put it in the `img/` folder.

Put your address / P.O. box / other info right below your picture. You can also disable any these elements by editing `profile` property of the YAML header of your `_pages/about.md`. Edit `_bibliography/papers.bib` and Jekyll will render your [publications page](/al-folio/publications/) automatically.

Link to your social media connections, too. This theme is set up to use [Font Awesome icons](http://fortawesome.github.io/Font-Awesome/) and [Academicons](https://jpswalsh.github.io/academicons/), like the ones below. Add your Facebook, Twitter, LinkedIn, Google Scholar, or just disable all of them.
 -->

<!-- I am currently a **Lead Research Scientist** at [Salesforce AI Research](https://www.salesforceairesearch.com/), where I lead a team building reliable and trustworthy enterprise AI agents. My research is fundamentally driven by the pursuit of **reliability and reasoning in LLMs**, specifically focusing on building reliable **agents** for long-horizon systems ([deep research](https://www.salesforce.com/blog/trusted-deepresearch/), [uncertainty quantification](https://arxiv.org/abs/2601.15703), [confidence calibration](https://arxiv.org/abs/2601.15778)), advancing multi-step **reasoning** via [post-training/RL](https://arxiv.org/abs/2509.25666) and [test-time scaling](https://aclanthology.org/2025.emnlp-industry.146/), and pushing the boundary of **self-improvement** capability (on-policy self-distillation). 

I am passionate about bridging the gap between frontier AI research and large-scale real-world impact. Prior to Salesforce, I was a **Senior Staff Research Scientist** and a founding member of the AI Research team at [Intuit](https://www.intuit.com/), where I architected industry-deployed hallucination detection and mitigation frameworks ([SAC$^3$](https://aclanthology.org/2023.findings-emnlp.1032/),[GAME](https://aclanthology.org/2025.findings-naacl.458/)), automatic prompt optimization libraries ([PhaseEvo](https://aclanthology.org/2025.acl-long.1431/),[SoS](https://aclanthology.org/2024.emnlp-industry.76/)), [reliable RAG systems](https://aclanthology.org/2024.emnlp-main.353/) ([Ski](https://aclanthology.org/2024.emnlp-main.1196/),[HyQE](https://aclanthology.org/2024.findings-emnlp.761/)), and post-training/alignment data collection pipelines ([IFML](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6c1843f11d34312b11ec5ff9a10c5a6-Abstract-Conference.html)) for enterprise finanical domain-specific LLM models, chatbots and agents. 

My technical roots lie deeply in extreme-scale computing. During my tenure as a **Staff Research Scientist** at [Oak Ridge National Laboratory (ORNL)](https://www.ornl.gov/), I architected distributed deep learning systems that scaled to 20,000+ GPUs (e.g., training ImageNet on ResNet50 in 10 minutes) on world-class supercomputers ([Summit](https://en.wikipedia.org/wiki/Summit_(supercomputer)), [Frontier](https://en.wikipedia.org/wiki/Frontier_(supercomputer))). As a Principal Investigator (PI)/co-PI, I led 7 DOE ASCR/ORNL projects (over 6.4 million in total) to pioneer "Generative AI for Science" to accelerate scientific simulations and experimental design in [Physics](https://www.sciencedirect.com/science/article/pii/S0264127519306859), [Chemistry](https://www.nature.com/articles/s41524-021-00554-0), and [Material Science](https://www.sciencedirect.com/science/article/abs/pii/S0079642525000738), with publications in top-tier journals ([40+ impact factor](https://www.sciencedirect.com/science/article/abs/pii/S0079642522000998) and [Nature series](https://www.nature.com/articles/s41524-021-00670-x)). I am a recipient of the "Promising EarlyCareer Researcher Award", from US Department of Energy. Before ORNL, I earned my Ph.D. from [Johns Hopkins University](https://www.jhu.edu/).

Beyond research, I am an active contributor to the open-source community, maintaining several heavily starred projects (3,000+ stars) focused on LLM RAG, Prompt Optimization, and Reliability.
 -->

I am a **Senior Staff Research Scientist (Research Lead)** at [Salesforce AI Research](https://blog.salesforceairesearch.com/), where I lead a team building **reliable, calibrated LLM models and self-evolving long-horizon AI agents**. My research turns **uncertainty, confidence, and consistency into first-class training signals** for post-training/RL, scalable evaluation, agent oversight, and self-evolving.

<em>I believe the next frontier of AI capability lies at the intersection of <strong>calibrated reasoning</strong> and <strong>self-improving agents</strong> — systems that know what they don't know and can autonomously improve through principled exploration.</em>

My current research focuses on:

- **Agentic Reinforcement Learning** — calibration-aware post-training, on-policy distillation, and self-evolving training environments (<a href="https://arxiv.org/abs/2604.16830">CaOPD</a>, <a href="https://arxiv.org/abs/2509.25666">NuRL</a>).
- **Alignment, Calibration & Honesty** — turning uncertainty and consistency into active training signals for honest, scalable LLM oversight (<a href="https://arxiv.org/abs/2601.15690">Passive→Active survey</a>, <a href="https://arxiv.org/abs/2601.15778">Agentic Confidence Calibration</a>).
- **Long-horizon Agents & Evaluation** — trajectory-level oversight and enterprise-scale agent benchmarks (<a href="https://arxiv.org/abs/2601.15703">Agentic Uncertainty Quantification</a>, <a href="https://www.salesforce.com/blog/trusted-deepresearch/">Trustworthy Deep Research</a>).

Previously, I was a Senior Staff Research Scientist and founding research lead at [Intuit AI Research](https://www.intuit.com/), for building reliable LLM systems, spanning LLM post-training, alignment, evaluation, and production deployment. I architected and deployed hallucination detection ([SAC3](https://arxiv.org/abs/2311.01740), used by 1,600+ internal users) and prompt optimization pipelines ([PhaseEvo](https://arxiv.org/abs/2402.11347), used by 2,000+ developers) for enterprise financial LLMs — recognized with the **Intuit CTO Award (top 1%)**. Earlier, as Staff Research Scientist at [Oak Ridge National Laboratory](https://www.ornl.gov/), I architected distributed deep learning at **20,000+ GPUs** on world-class supercomputers ([Summit](https://en.wikipedia.org/wiki/Summit_(supercomputer)), [Frontier](https://en.wikipedia.org/wiki/Frontier_(supercomputer))) and led 7 DOE projects ($6.4M total) on Generative AI for Science, recognized with the **DOE Promising Early-Career Researcher Award**. I earned my Ph.D. at [Johns Hopkins University](https://www.jhu.edu/).


<!-- 
 I study interpretable human-AI interaction for computer vision and machine autonomy. I am also interested in understanding various human-centric properties of current AI models beyond their accuracy, such as <a href="http://cnnlocalization.csail.mit.edu/">explainability</a>, <a href="http://netdissect.csail.mit.edu/">interpretability</a>, <a href="https://genforce.github.io/higan/">steerability</a>, <a href="https://metadriverse.github.io/metadrive/">generalization</a>, and <a href="https://decisionforce.github.io/HACO/">safety</a>. Some of the earlier works I co-authored are <a href="http://cnnlocalization.csail.mit.edu/">Class Activation Mapping (CAM)</a>, <a href="http://places2.csail.mit.edu/">Places</a>, <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a>, <a href="http://netdissect.csail.mit.edu/">Network Dissection</a>. 

See <a href="https://metadriverse.github.io/">MetaDriverse</a> for recent work on machine autonomy and <a href="https://genforce.github.io/">GenForce</a> for recent work on generative modeling. -->
