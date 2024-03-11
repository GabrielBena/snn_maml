# Source: https://mousebender.wordpress.com/2006/11/10/recursive-getattrsetattr/


def rec_getattr(obj, attr):
    """Get object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_getattr(a, 'b.c')
    4
    """
    if "." not in attr:
        return getattr(obj, attr)
    else:
        L = attr.split(".")
        return rec_getattr(getattr(obj, L[0]), ".".join(L[1:]))


def rec_setattr(obj, attr, value):
    """Set object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_setattr(a, 'b.c', 2)
    >>> a.b.c
    2
    """
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        L = attr.split(".")
        rec_setattr(getattr(obj, L[0]), ".".join(L[1:]), value)
