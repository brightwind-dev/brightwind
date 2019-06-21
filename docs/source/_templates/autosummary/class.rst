{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. autosummary::
        :toctree: {{ objname }}
        {% if methods %}
        {% for item in methods %}
        {{ objname }}.{{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}