from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    """Multiplies the value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value