{% if top_line %}
## {{ top_line }}
{% endif %}
{% for section in sections %}
{% if sections[section] %}
{% for category, val in definitions.items() if category in sections[section] and category != 'trivial' %}

### {{ definitions[category]['name'] }}
{% if definitions[category]['showcontent'] %}
{% for text, values in sections[section][category]|dictsort(by='value') %}
{% set issue = values|first %}
{% set issue_number = issue.split('#')[1].split(']')[0] %}
{% if 'pr.' in issue %}
- {{ text }} [#{{ issue_number.split('.')[-1] }}](https://github.com/lmmx/page-dewarp/pull/{{ issue_number.split('.')[-1] }})
{% elif 'co.' in issue %}
- {{ text }} [{{ issue_number.split('.')[-1] }}](https://github.com/lmmx/page-dewarp/commit/{{ issue_number.split('.')[-1] }})
{% else %}
- {{ text }} {{ values|sort|join(',\n  ') }}
{% endif %}

{% endfor %}
{% else %}
- {{ sections[section][category]['']|sort|join(', ') }}
{% endif %}
{% if sections[section][category]|length == 0 %}
No significant changes.
{% endif %}
{% endfor %}
{% else %}
No significant changes.
{% endif %}
{% endfor %}

{# This comment and newline ensures the file ends with a newline #}
