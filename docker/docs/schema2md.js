const fs = require('fs/promises');
const { join, basename } = require('path');

async function parseFile(filepath) {
  console.log(`Parsing file ${filepath}...`);
  const schemaName = basename(filepath, '.schema.json').replace(/_/g, '.');
  const schema = JSON.parse(await fs.readFile(filepath, 'utf8'));

  const result = [];

  result.push(`### ${schemaName}[^](#schema-reference)`);
  if (schema.description) {
    result.push(schema.description);
  }
  if (schema.examples && schema.examples.length) {
    if (schema.examples.length > 1) {
      result.push('#### Examples');
    } else {
      result.push('#### Example');
    }
    schema.examples.forEach(example => {
      result.push('```json\n' + JSON.stringify(example, undefined, 2) + '\n```');
    });
  }

  const propertyStack = [{ name: schemaName, properties: schema.properties, required: schema.required }];

  while (propertyStack.length) {
    result.push(...parseProperties(propertyStack.shift(), propertyStack));
  }

  console.log(`    ...done`);
  return {
    schemaName,
    md: result.join('\n\n')};
}

function parseProperties({ name, properties = {}, required = []}, propertyStack) {
  const props = Object.entries(properties).map(([key, value]) => {
    const prp = `\`${key}\` ${required.includes(key) ? '_required_ ' : ''}(\`${value.type}\`)`;
    const description = [
      value.description
    ];
    if (value.enum) {
      description.push(
        '**Possible values:**<ul>' + value.enum.map(v => `<li>\`${v}\`</li>`).join('') + '</ul>'
      );
    }
    if (value.const) {
      description.push(`Value must be \`${value.const}\`.`);
    }
    if (value.examples) {
      description.push(
        `**Example${value.examples.length > 1 ? 's' : ''}:**<br/>` + value.examples.map(v => `\`${v}\``).join('<br/>')
      );
    }
    if (value.type === 'object') {
      propertyStack.push({
        name: `${name}.${key}`,
        properties: value.properties,
        required: value.required
      });
      description.push(`_Details [below](#${name.replace(/\./g, '')}${key}-properties)._`);
    }

    return `| ${prp} | ${description.join('<br/><br/>')} |`;
  });
  return [
    `#### \`${name}\` properties`,
    `| Property | Description |\n|---|---|\n${props.join('\n')}`
  ]
}

function prepareSection(type, mds) {
  return `## ${type[0].toUpperCase() + type.slice(1)}\n\n` +
    mds.map(({schemaName}) => `1. [${schemaName}](#${schemaName.replace(/\./g, '')})`).join('\n') +
    `\n\n${mds.map(e => e.md).join('\n\n<hr/>\n\n')}`;
}

async function generateOutput(mds, path) {
  const result = [
  ];
  const toc = [];
  for(const type of ['events', 'metrics', 'other']) {
    console.log(`Will generate output for ${type}...`);
    if (mds[type].length) {
      const typeName = type[0].toUpperCase() + type.slice(1);
      toc.push(`1. [${typeName}](#${type})`);
      result.push(prepareSection(type, mds[type]));
      console.log('    ...done');
    } else {
      console.log('    ...no entries to process');
    }
  };
  if (toc.length) {
    toc.push('\n'); //join will add two new lines at end
  }

  await fs.writeFile(
    join(path, `schema-reference.md`),
    `---\ntitle: Schema Reference\n---\n\n# Schema Reference\n\n${toc.join('\n')}${result.join('\n\n')}`,
    'utf8');
}

async function main(tempDir, outPath) {
  const output = {
    events: [],
    metrics: [],
    other: [],
  }

  const files = (await fs.readdir(tempDir, 'utf8')).filter(f => f.endsWith('.schema.json'));
  for (const filename of files) {
    const section = await parseFile(join(tempDir, filename));
    if (filename.startsWith('event_')) {
      output.events.push(section);
    } else if (filename.startsWith('metric_')) {
      output.metrics.push(section);
    } else {
      output.other.push(section);
    }
  };
  
  await generateOutput(output, outPath);
}
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.log('Usage: node schema2md.js <source_dir> <out_dir>');
  process.exit(1);
}
main(args[0], args[1]).catch(err => {
  console.error(err);
  process.exit(1);
});
