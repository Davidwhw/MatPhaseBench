specified_dimension_description_prompt = """
You are a materials science expert specializing in phase diagram interpretation.
Your task is to describe a phase diagram image from a materials science paper as a materials science expert.

The image may be a binary phase diagram, ternary phase diagram, isothermal section, vertical section, liquidus projection, liquidus surface, or element-rich region.

## Selected dimensions:
You must describe the image only from the selected dimensions listed below:

{{SELECTED_DIMENSIONS}}

## Dimension definitions:

{{DIMENSION_DEFINITIONS}}

Rules:

1. For each selected dimension, provide a complete and detailed description as much as possible.
2. After describing each selected dimension separately, generate one comprehensive description that summarizes and integrates the information from all selected dimensions.
3. The comprehensive description must be based only on the selected dimensions and must not introduce information that is not visible in the image.
4. Preserve chemical formulas, phase labels, Greek letters, subscripts, superscripts, and mathematical symbols as faithfully as possible. Use LaTeX-style notation whenever possible, for example: Al$_2$O$_3$, Ni$_5$Zr, $\alpha$, $\beta$, $\gamma$, $\alpha$-Ti, L + $\beta$, and T$_m$.
5. Never output an empty description for any selected dimension. For every selected dimension, make the best possible description strictly from visible image evidence. If the dimension cannot be fully determined, describe the relevant observable evidence, partial information, uncertainty, or limitation instead of leaving the field empty. Do not hallucinate information that is not visible in the image.
6. Finally, generate a comprehensive_description that fully summarizes the selected dimensions without compressing or omitting important visible information.
7. Output only valid JSON. Do not include any explanation outside the JSON.

Return JSON in the following schema:

{{OUTPUT_SCHEMA}}
"""


DIMENSION_DEFINITIONS = {
    "system_scope": (
        "## Phase Diagram System\n"
        "Describe the material system shown in the phase diagram, such as binary, "
        "ternary, or multicomponent system. Identify the main elements, compounds, "
        "or alloy system involved."
    ),

    "diagram_type": (
        "## Phase Diagram Type\n"
        "Describe the diagram type and graphical form, such as isothermal section, "
        "vertical section, liquidus projection, liquidus surface, or combined diagram. "
        "Also describe the visible axes, units, and overall layout when available."
    ),

    "diagram_completeness": (
        "## Phase Diagram Completeness\n"
        "Describe whether the phase diagram is complete or partial. Explain whether "
        "the diagram covers the full composition range of the material system or only "
        "a specific composition interval, subsystem, or element-rich region."
    ),

    "phase_regions_boundaries": (
        "## Phase Region and Phase Boundary Identification\n"
        "Describe the visible phase regions and phase boundaries, including single-phase "
        "regions, two-phase regions, three-phase regions, and their topological relations. "
        "Identify important boundaries such as liquidus, solidus, solvus, miscibility-gap "
        "boundaries, and invariant horizontal lines when visible."
    ),

    "invariant_reactions": (
        "## Invariant Reactions\n"
        "Describe visible or directly indicated invariant reactions, including congruent "
        "melting, eutectic, eutectoid, monotectic, metatectic, peritectic, peritectoid, "
        "and syntectic reactions. For each reaction, describe the reaction type, "
        "temperature, composition, participating phases, and local phase-boundary topology.\n"

        "Use the following reaction definitions as guidance:\n"

        "- **Congruent Melting**: a liquid transforms directly into a single solid phase, "
        "or a solid melts directly into a liquid with the same composition. The reaction "
        "has one phase on each side, such as L -> A or A -> L. Its topology is a continuous "
        "solidification or melting relation between the liquid region and the solid region, "
        "usually bounded by liquidus and solidus lines with an L + A two-phase region.\n"

        "- **Eutectic Reaction**: one liquid transforms into two solid phases at a fixed "
        "temperature, written as L -> A + B. Its topology usually shows two liquidus "
        "boundaries converging to a eutectic point, with an A + B two-phase region below "
        "a horizontal eutectic isotherm.\n"

        "- **Eutectoid Reaction**: one solid phase decomposes into two different solid phases "
        "at a fixed temperature, written as A -> B + C. Its topology usually shows a "
        "high-temperature single solid phase ending at the eutectoid point, with a B + C "
        "two-phase region below a horizontal eutectoid isotherm.\n"

        "- **Monotectic Reaction**: one liquid decomposes into another liquid and one solid "
        "phase at a fixed temperature, written as L1 -> L2 + A. Its topology involves "
        "L1, L2, and A phase regions meeting near a monotectic isotherm, often related "
        "to liquid immiscibility.\n"

        "- **Metatectic Reaction**: one solid phase transforms into a liquid and another "
        "solid phase on heating, written as A -> L + B. Its topology shows a single "
        "solid phase ending at a metatectic isotherm and connecting to an L + B "
        "two-phase region.\n"

        "- **Peritectic Reaction**: one liquid and one solid phase react to form a new solid "
        "phase at a fixed temperature, written as L + A -> B. Its topology usually shows "
        "the L + A two-phase region ending at a peritectic isotherm and connecting to "
        "the newly formed single solid phase B.\n"

        "- **Peritectoid Reaction**: two solid phases react to form another solid phase at "
        "a fixed temperature, written as A + B -> C. Its topology shows the A + B "
        "two-phase region ending at a peritectoid isotherm and connecting to the newly "
        "formed single solid phase C.\n"

        "- **Syntectic Reaction**: two different liquids react to form one solid phase at "
        "a fixed temperature, written as L1 + L2 -> A. Its topology involves L1, L2, "
        "and solid A meeting near a syntectic isotherm, representing two liquid phases "
        "jointly forming a solid phase.\n"
    )
}
