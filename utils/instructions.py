# class SPEERPromptTemplate:
#     Non_Guided="""Generate the BRIEF HOSPITAL COURSE summary.
# {source}
# BRIEF HOSPITAL COURSE:"""

#     Guided="""Generate the BRIEF HOSPITAL COURSE summary using only the medical entities provided.
# {source}
# ### ENTITIES
# {entity}
# BRIEF HOSPITAL COURSE:"""

#     SPEER="""Retrieve a subset of the medical entities in double brackets {{{{}}}} and use them to generate the BRIEF HOSPITAL COURSE summary.
# {source}
# BRIEF HOSPITAL COURSE:"""

class SPEER:
    Non_Guided="""Generate the BRIEF HOSPITAL COURSE summary."""

    Guided="""Generate the BRIEF HOSPITAL COURSE summary using only the medical entities provided."""

    SPEER="""Retrieve a subset of the medical entities in double brackets {{}} and use them to generate the BRIEF HOSPITAL COURSE summary."""