const fs = require('fs/promises');
const { join, basename } = require('path');

async function parseFile(filepath) {
  console.log(`Parsing file ${filepath}...`);
  const schemaName = basename(filepath, '.schema.json').replace(/^(event|metric)_/, '').replace(/_/g, '.');
  const schema = JSON.parse(await fs.readFile(filepath, 'utf8'));

  const result = [];

  result.push(`## ${schemaName}`);
  if (schema.description) {
    result.push(schema.description);
  }
  if (schema.examples && schema.examples.length) {
    if (schema.examples.lengh > 1) {
      result.push('### Examples');
    } else {
      result.push('### Example');
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
  return result.join('\n\n');
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
    `### \`${name}\` properties`,
    `| Property | Description |\n|---|---|\n${props.join('\n')}`
  ]
}

async function generateOutput(mds, path) {
  for(const type of ['events', 'metrics', 'other']) {
    console.log(`Will generate output for ${type}...`);
    if (mds[type].length) {
      await fs.writeFile(
        join(path, `${type}.md`),
        `---\ntitle: ${type[0].toUpperCase() + type.slice(1)}\n---\n\n${mds[type].join('\n\n')}`,
        'utf8');
      console.log('    ...done');
    } else {
      console.log('    ...no entries to process');
    }
  };
}

async function main(tempDir, outPath) {
  const output = {
    events: [],
    metrics: [],
    other: [],
  }

  const files = (await fs.readdir(tempDir, 'utf8')).filter(f => f.endsWith('.schema.json'));
  for (const filename of files) {
    const md = await parseFile(join(tempDir, filename));
    if (filename.startsWith('event_')) {
      output.events.push(md);
    } else if (filename.startsWith('metric_')) {
      output.metrics.push(md);
    } else {
      output.other.push(md);
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
