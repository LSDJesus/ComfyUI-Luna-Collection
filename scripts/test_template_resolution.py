"""Test template resolution for YAML wildcards"""

import sys
sys.path.insert(0, 'D:/AI/ComfyUI/custom_nodes/ComfyUI-Luna-Collection/nodes')
from luna_yaml_wildcard import LunaYAMLWildcardParser

parser = LunaYAMLWildcardParser('D:/AI/SD Models/wildcards_atomic')

# Test 1: {body} should use a template from templates.full/descriptive
print('Test 1: {body} template selection')
for i in range(3):
    result = parser.process_prompt('{body}', seed=100+i)
    print(f'  {i+1}: {result}')

# Test 2: {body:hair} - should look for templates.hair first, then items
print()
print('Test 2: {body:hair} path selection')
for i in range(3):
    result = parser.process_prompt('{body:hair}', seed=200+i)
    print(f'  {i+1}: {result}')

# Test 3: Deep path selection
print()
print('Test 3: Direct path selections')
print(f'  skin.tone.pale: {parser.process_prompt("{body:skin.tone.pale}", seed=1)}')
print(f'  hair.color.fantasy: {parser.process_prompt("{body:hair.color.fantasy}", seed=2)}')
print(f'  eyes.color.fantasy: {parser.process_prompt("{body:eyes.color.fantasy}", seed=3)}')

# Test 4: Inline templates with multiple files
print()
print('Test 4: Multi-file composition')
template = """
{composition:shot_type.distance}, {composition:angle.vertical}, 
{body: a [body_type.types] woman with [skin.tone.medium] skin, [hair.color.natural] [hair.length] hair}, 
{pose:posture.type.standing.casual},
{clothing:tops.types.casual},
{setting:location.indoor.residential},
{lighting:type.natural}
"""
result = parser.process_prompt(template.strip(), seed=42)
print(f'  Result: {result}')

# Test 5: Using file-level templates (no path)
print()
print('Test 5: File-level templates')
print(f'  {{clothing}}: {parser.process_prompt("{clothing}", seed=50)}')
print(f'  {{pose}}: {parser.process_prompt("{pose}", seed=51)}')
print(f'  {{lighting}}: {parser.process_prompt("{lighting}", seed=52)}')

# Test 6: Full prompt composition
print()
print('Test 6: Full realistic prompt')
full_prompt = """{composition:shot_type.distance.medium}, {composition:angle.vertical},
{body: beautiful [ethnicity.options] woman, [body_type.types] figure, [skin.tone] skin, [hair.length] [hair.color.natural] [hair.style.texture] hair, [eyes.color.natural] eyes},
{expression:emotion.positive.happy} expression,
{clothing: wearing [tops.types.casual] and [bottoms.types.pants]},
{pose:posture.type.standing.casual},
{setting: in a [location.indoor.residential]},
{lighting:type.natural.sunlight}"""

for i in range(3):
    result = parser.process_prompt(full_prompt, seed=300+i)
    print(f'  {i+1}: {result}')
    print()

# Test 7: Random number generation
print('Test 7: Random number generation')
print(f'  {{1-10}} integers:')
for i in range(5):
    result = parser.process_prompt('{1-10}', seed=400+i)
    print(f'    {result}', end=' ')
print()

print(f'  {{0.5-1.5:0.1}} floats with 0.1 resolution:')
for i in range(5):
    result = parser.process_prompt('{0.5-1.5:0.1}', seed=500+i)
    print(f'    {result}', end=' ')
print()

print(f'  {{1-100:5}} integers with step 5:')
for i in range(5):
    result = parser.process_prompt('{1-100:5}', seed=600+i)
    print(f'    {result}', end=' ')
print()

print(f'  {{0.01-0.99:0.01}} fine resolution:')
for i in range(5):
    result = parser.process_prompt('{0.01-0.99:0.01}', seed=700+i)
    print(f'    {result}', end=' ')
print()

# Test 8: Mixed prompt with numbers
print()
print('Test 8: Mixed prompt with random numbers')
mixed = "CFG: {4-12}, denoise: {0.3-0.8:0.05}, steps: {20-50:5}"
for i in range(3):
    result = parser.process_prompt(mixed, seed=800+i)
    print(f'  {i+1}: {result}')
