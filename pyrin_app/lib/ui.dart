import "dart:ui";

import "package:flutter/cupertino.dart";
import "package:flutter/material.dart";
import "package:flutter/services.dart";
import "package:flutter_svg/svg.dart";
import "package:mobile_scanner/mobile_scanner.dart";
import "package:pyrin_app/core/addressbook.dart";

class PyrinColors
{
    static const Color BACKGROUND_COLOR = Color(0xff101212);
    static const Color TEXT_COLOR = PyrinColors.WHITE_COLOR;
    static const Color LIGHT_TEXT_COLOR = Color(0xccfafafa);
    static const Color WHITE_COLOR = Color(0xfffafafa);
    static const Color RED_COLOR = Color(0xffee3232);
    static const Color ORANGE_COLOR = Color(0xfffa9a2a);
    static const Color GREEN_COLOR = Color(0xff52d377);

    static const Color BLACK1_COLOR = Color(0xff1c1d21);

    static const Gradient WHITE_GRADIENT = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [
        Colors.white,
        Color(0xFFEDE5EE),
      ],
    );
}

class PyrinIconButton
{
    final String icon;
    final VoidCallback? onClick;
    final Color? color;

    PyrinIconButton({required this.icon, this.onClick, this.color});
}

class PyrinFlatButton extends StatelessWidget
{
    final String text;
    final String? leftIcon;
    final String? rightIcon;
    final VoidCallback onClick;

    PyrinFlatButton({Key? key, required this.text, this.leftIcon, this.rightIcon, required this.onClick});

    @override
    Widget build(BuildContext context)
    {
        return TextButton(
          onPressed: onClick,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (leftIcon != null)
                Container(
                  margin: EdgeInsets.only(right: 10),
                  child: SvgPicture.asset("assets/icons/$leftIcon.svg"),
                ),
              Text(
                  text,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodySmall!.copyWith(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: PyrinColors.TEXT_COLOR
                  )
              ),
              if (rightIcon != null)
                Container(
                  margin: EdgeInsets.only(left: 10),
                  child: SvgPicture.asset("assets/icons/$rightIcon.svg"),
                ),
            ],
          ),
        );
    }
}

class PyrinElevatedButton extends StatefulWidget
{
    final bool wide;
    final bool disabled;
    final bool secondary;
    final String text;
    final Function onClick;
    final double? width;

    PyrinElevatedButton({
      Key? key,
      required this.text,
      required this.onClick,
      this.wide = false,
      this.disabled = false,
      this.secondary = false,
      this.width,
});

  @override
  State<PyrinElevatedButton> createState() => _PyrinElevatedButtonState();
}

class _PyrinElevatedButtonState extends State<PyrinElevatedButton> {
    bool _loader = false;

    @override
    Widget build(BuildContext context)
    {
        final Color textColor =
          widget.disabled
              ? PyrinColors.LIGHT_TEXT_COLOR
              : !widget.secondary ? PyrinColors.BACKGROUND_COLOR : PyrinColors.WHITE_COLOR;

        return Container(
            width: widget.width != null ? widget.width : widget.wide ? double.infinity : null,
            decoration: BoxDecoration(
              gradient: !widget.disabled ? PyrinColors.WHITE_GRADIENT : null,
              color: !widget.disabled ? null : PyrinColors.BLACK1_COLOR,
              borderRadius: BorderRadius.circular(68.0),
              border: !widget.secondary ? null : Border.all(
                color:  PyrinColors.BLACK1_COLOR,
                width: 1,
              ),
            ),
            child: Container(
              margin: !widget.secondary ? null : EdgeInsets.all(1),
              decoration: !widget.secondary ? null : BoxDecoration(
                borderRadius: BorderRadius.circular(68.0),
              ),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  foregroundColor: textColor,
                  backgroundColor: !widget.secondary ? Colors.transparent : PyrinColors.BACKGROUND_COLOR,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(68.0),
                  ),
                  padding: widget.wide ? EdgeInsets.all(24) : EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                  elevation: 0,
                  shadowColor: Colors.transparent,
                ),
                onPressed: !widget.disabled ? _onClick : null,
                child: _loader ? SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation<Color>(textColor),
                  ),
                ) : Text(widget.text, style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                    fontWeight: FontWeight.w500,
                    color: textColor
                )),
              ),
            ),
          );
    }

    _onClick() async
    {
        if (_loader)
          return;

        setState(() => _loader = true);

        await widget.onClick();

        setState(() => _loader = false);
    }
}

class PyrinTextField extends StatelessWidget
{
    final String? name;
    final String hintText;
    final PyrinIconButton? iconButton;
    final TextEditingController? controller;
    final int? maxLines;
    final bool? readOnly;
    final bool? obscureText;
    final TextInputType? keyboardType;

    PyrinTextField({
      Key? key,
      this.name,
      this.hintText = "",
      this.iconButton,
      this.controller,
      this.maxLines,
      this.readOnly,
      this.obscureText,
      this.keyboardType,
    });

    @override
    Widget build(BuildContext context)
    {
        return PyrinGroup(
            label: name,
            child: iconButton != null ? Stack(
              alignment: Alignment.centerLeft,
              children: [
                _buildTextField(context),
                Align(
                  alignment: Alignment.centerRight,
                  child: Container(
                    margin: EdgeInsets.only(right: 10),
                    // color: Colors.red,
                    child: IconButton(
                      onPressed: iconButton!.onClick,
                      icon: SvgPicture.asset("assets/icons/${iconButton!.icon}.svg", color: iconButton!.color ?? PyrinColors.LIGHT_TEXT_COLOR),
                    ),
                  ),
                ),
              ],
            ) : _buildTextField(context),
        );
    }

    Widget _buildTextField(BuildContext context)
    {
      return TextField(
          controller: controller,
          maxLines: maxLines ?? 1,
          readOnly: readOnly ?? false,
          obscureText: obscureText ?? false,
          style: Theme.of(context).textTheme.bodyMedium,
          keyboardType: keyboardType,
          decoration: InputDecoration(
            filled: true,
            fillColor: PyrinColors.BLACK1_COLOR,
            border: OutlineInputBorder(
              borderSide: BorderSide.none,
              borderRadius: BorderRadius.all(Radius.circular(8)),
            ),
            contentPadding: EdgeInsets.fromLTRB(18.0, 18.0, iconButton != null ? (18.0 + 36) : 18.0, 18.0),
            hintStyle: Theme.of(context).textTheme.bodyMedium!.copyWith(color: PyrinColors.LIGHT_TEXT_COLOR),
            hintText: hintText,
          ),
        );
    }
}

class PyrinPasswordTextField extends StatefulWidget
{
    final String? name;
    final String hintText;
    final TextEditingController? controller;

    PyrinPasswordTextField({
      Key? key,
      this.name,
      this.hintText = "",
      this.controller,
    });

    @override
    State<PyrinPasswordTextField> createState() => _PyrinPasswordTextFieldState();
}

class _PyrinPasswordTextFieldState extends State<PyrinPasswordTextField>
{
    bool _obscureText = true;

    @override
    Widget build(BuildContext context)
    {
        return PyrinTextField(
          name: widget.name,
          hintText: widget.hintText,
          controller: widget.controller,
          obscureText: _obscureText,
          iconButton: PyrinIconButton(
            icon: "eye",
            color: _obscureText ? null : PyrinColors.TEXT_COLOR,
            onClick: () => setState(() => _obscureText = !_obscureText),
          ),
        );
    }
}

class PyrinDropdownItem
{
    final String value;
    final Widget child;

    const PyrinDropdownItem({required this.value, required this.child});
}

class PyrinDropdown extends StatefulWidget
{
    final String? value;
    final List<String> items;
    final Widget Function(BuildContext context, String value, Widget? child) builder;

    PyrinDropdown({Key? key, this.value, required this.items, required this.builder});

    @override
    State<PyrinDropdown> createState() => _PyrinDropdownState();
}

class _PyrinDropdownState extends State<PyrinDropdown>
{
    late String dropdownValue;

    @override
    void initState()
    {
        super.initState();

        dropdownValue = widget.value ?? widget.items.first;
    }

    @override
    Widget build(BuildContext context)
    {
      return DropdownButton(
        value: dropdownValue,
        icon: SvgPicture.asset("assets/icons/arrow-down.svg", color: PyrinColors.TEXT_COLOR),
        underline: Container(),
        style: Theme.of(context).textTheme.bodyMedium!.copyWith(
          fontWeight: FontWeight.w500,
          fontSize: 14,
          color: PyrinColors.TEXT_COLOR,
        ),
        dropdownColor: PyrinColors.BLACK1_COLOR,
        borderRadius: BorderRadius.circular(12),
        items: widget.items.map<DropdownMenuItem<String>>((value)
        {
          return DropdownMenuItem<String>(
            value: value,
            child: widget.builder(context, value, Container(
              margin: const EdgeInsets.all(8),
              child: Text(value, style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                fontWeight: FontWeight.w600,
                fontSize: 16,
                color: PyrinColors.TEXT_COLOR,
              )),
            )),
          );
        }).toList(),
        onChanged: (String? value)
        {
          setState(() {
            dropdownValue = value!;
          });
        },
      );
    }

    @override
    void didUpdateWidget(PyrinDropdown oldWidget)
    {
        super.didUpdateWidget(oldWidget);

        if (oldWidget.value != widget.value)
        {
            setState(()
            {
                dropdownValue = widget.value ?? widget.items.first;
            });
        }
    }
}


class PyrinTabItem
{
    final String name;
    final String value;

    PyrinTabItem({required this.name, required this.value});
}

class PyrinTabs extends StatefulWidget
{
    final List<PyrinTabItem> items;
    final Function(String value) onChange;

    PyrinTabs({Key? key, required this.items, required this.onChange});

    @override
    State<PyrinTabs> createState() => _PyrinTabsState();
}

class _PyrinTabsState extends State<PyrinTabs>
{
    String? _current;

    @override
    Widget build(BuildContext context)
    {
        final current = _current ?? widget.items.first.value;

        return PyrinCard(
            borderRadius: 26,
            padding: EdgeInsets.all(6),
            shrink: true,
            child: Container(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.end,
                children: widget.items.map((item)
                {
                    return ElevatedButton(
                      onPressed: ()
                      {
                          setState(() => _current = item.value);
                          widget.onChange(item.value);
                      },
                      style: ElevatedButton.styleFrom(
                        elevation: 0,
                        shadowColor: Colors.transparent,
                        backgroundColor: current == item.value ? PyrinColors.WHITE_COLOR : Colors.transparent,
                        padding: EdgeInsets.symmetric(vertical: 12, horizontal: 24),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(26),
                        ),
                      ),
                      child: Text(
                        item.name,
                        style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                          fontWeight: FontWeight.w500,
                          fontSize: 14,
                          color: current == item.value ? PyrinColors.BACKGROUND_COLOR : PyrinColors.TEXT_COLOR.withOpacity(0.4)
                        )
                      ),
                    );

                    return Container(
                      padding: EdgeInsets.symmetric(vertical: 12, horizontal: 24),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(26),
                        color: current == item.value ? PyrinColors.WHITE_COLOR : Colors.transparent,
                      ),
                      child: Text(
                          item.name,
                          style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                              fontWeight: FontWeight.w500,
                              fontSize: 14,
                              color: current == item.value ? PyrinColors.BACKGROUND_COLOR : PyrinColors.TEXT_COLOR.withOpacity(0.4)
                          )
                      ),
                    );
                }).toList(),
              ),
            )
        );
    }
}

class PyrinGroup extends StatelessWidget
{
    final String? label;
    final String? footer;
    final bool small;
    final Widget child;

    PyrinGroup({Key? key, required this.label, required this.child, this.small = false, this.footer});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ...label != null ? [
                Container(
                    padding: small ? null : EdgeInsets.symmetric(horizontal: 10),
                    child: Text(label!, style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                      fontWeight: FontWeight.w400,
                      fontSize: small ? 12: 14,
                    ))
                ),
                SizedBox(height: 8),
              ] : [],
              child,
              ...footer != null ? [
                Container(
                    padding: EdgeInsets.all(10),
                    child: Text(footer!, style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                      fontWeight: FontWeight.w400,
                      fontSize: 11,
                    ))
                ),
                SizedBox(height: 8),
              ] : [],
            ],
          ),
        );
    }
}

class PyrinSwitch extends StatefulWidget
{
  PyrinSwitch({Key? key});

  @override
  State<PyrinSwitch> createState() => _PyrinSwitchState();
}

class _PyrinSwitchState extends State<PyrinSwitch> {
    bool switchValue = false;

    @override
    Widget build(BuildContext context)
    {
        return Theme(
          data: ThemeData(
            useMaterial3: true
          ).copyWith(
            colorScheme: Theme.of(context).colorScheme.copyWith(
              outline: Colors.transparent,
            ),
          ),
          child: Switch(
            value: switchValue,
            onChanged: (bool value)
            {
              setState(()
              {
                  switchValue = value;
              });
            },
            thumbColor: MaterialStateProperty.resolveWith<Color>((Set<MaterialState> states) {
              if (states.contains(MaterialState.disabled)) {
                return Colors.grey.shade400;
              }
              if (states.contains(MaterialState.selected)) {
                return PyrinColors.TEXT_COLOR;
              }
              return PyrinColors.TEXT_COLOR;
            }),
            trackColor: MaterialStateProperty.resolveWith<Color>((Set<MaterialState> states) {
              if (states.contains(MaterialState.disabled)) {
                return Colors.grey.shade200;
              }
              if (states.contains(MaterialState.selected)) {
                return PyrinColors.TEXT_COLOR.withOpacity(0.5);
              }
              return PyrinColors.TEXT_COLOR.withOpacity(0.2);
            }),
            overlayColor: MaterialStateProperty.resolveWith<Color?>((Set<MaterialState> states) {
              return Colors.transparent;
            }),
          ),
        );
    }
}

class PyrinCardGroup extends StatelessWidget
{
    final String name;
    final String text;
    final Widget? leading;

    PyrinCardGroup({Key? key, required this.name, required this.text, this.leading});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          padding: EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: PyrinColors.BLACK1_COLOR,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                name,
                style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                  fontWeight: FontWeight.w600,
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 8),
              Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  ConstrainedBox(
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.6,
                    ),
                    child: Text(
                      text,
                      style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                        fontSize: 12,
                        color: PyrinColors.TEXT_COLOR.withOpacity(0.5),
                      ),
                    ),
                  ),
                  if (leading != null)
                    leading!,
                ],
              ),
            ],
          ),
        );
    }
}

class PyrinTitle extends StatelessWidget
{
  final String text;

  PyrinTitle({Key? key, required this.text});

  @override
  Widget build(BuildContext context)
  {
    return Container(
      margin: EdgeInsets.only(bottom: 20),
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: Theme.of(context).textTheme.bodyLarge!.copyWith(
          fontWeight: FontWeight.w600,
          fontSize: 24,
        ),
      ),
    );
  }
}

class PyrinSubtitle extends StatelessWidget
{
  final String text;

  PyrinSubtitle({Key? key, required this.text});

  @override
  Widget build(BuildContext context)
  {
    return Container(
      margin: EdgeInsets.only(bottom: 20),
      child: Text(
        text,
        textAlign: TextAlign.center,
        style: Theme.of(context).textTheme.bodyMedium!.copyWith(
          fontWeight: FontWeight.w500,
          fontSize: 14,
          color: PyrinColors.TEXT_COLOR.withOpacity(0.4),
        ),
      ),
    );
  }
}

// TODO: We could work more on the design
class PyrinCard extends StatelessWidget
{
    final Widget child;
    final double borderRadius;
    final EdgeInsets padding;
    final bool shrink;

    PyrinCard({
      Key? key,
      required this.child,
      this.borderRadius = 12,
      this.padding = const EdgeInsets.all(12),
      this.shrink = false
    });

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: shrink ? null : double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(borderRadius),
            color: Colors.white.withOpacity(0.02),
            boxShadow: [
              BoxShadow(
                color: Color.fromRGBO(255, 255, 255, 0.05),
                offset: Offset(0, -1),
                blurRadius: 184,
                spreadRadius: 0,
              ),
              BoxShadow(
                // color: Color.fromRGBO(255, 255, 255, 0.29),
                color: Color.fromRGBO(255, 255, 255, 0.15),
                offset: Offset(0, -1),
                blurRadius: 1,
                spreadRadius: 0,
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(borderRadius),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 17.5, sigmaY: 17.5),
              child: Container(
                padding: padding,
                color: Color(0xff0e0e0e),
                child: child,
              ),
            ),
          ),
        );
    }
}

// TODO: Use in the buttons ?
class PyrinCircleIcon extends StatelessWidget
{
    final Widget child;
    final bool darker;

    PyrinCircleIcon({Key? key, required this.child, this.darker = false});

    @override
    Widget build(BuildContext context)
    {
        final double size = 52;

        return Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(size),
            color: Colors.white.withOpacity(0.02),
            border: Border.all(
              color: Color.fromRGBO(16, 18, 18, 0.05),
              width: 0.5,
            ),
            boxShadow: [
              BoxShadow(
                color: Color.fromRGBO(255, 255, 255, 0.15),
                offset: Offset(0, -1),
                blurRadius: 0,
                spreadRadius: 0,
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(size),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 51, sigmaY: 51),
              child: Container(
                color: darker ? PyrinColors.BACKGROUND_COLOR : Colors.white.withOpacity(0.02),
                child: Center(
                  child: child,
                ),
              ),
            ),
          ),
        );
    }
}

class CircleIconButton extends StatelessWidget
{
    final String icon;
    final String? text;
    final VoidCallback onClick;

    const CircleIconButton({super.key, required this.icon, this.text, required this.onClick});

    @override
    Widget build(BuildContext context)
    {
        return Column(
            children: [
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.02),
                  borderRadius: BorderRadius.circular(60),
                  border: Border(
                    top: BorderSide(color: Colors.white.withOpacity(0.1), width: 1),
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.white.withOpacity(0.15),
                      offset: Offset(0, 0.25), // TODO: ?
                      blurRadius: 0,
                      spreadRadius: 0,
                    ),
                  ],
                ),
                // child: SvgPicture.asset("assets/icons/swap.svg", width: 32, height: 32),
                // child: BackdropFilter(
                //   filter: ImageFilter.blur(sigmaX: 51, sigmaY: 51),
                //   child: SvgPicture.asset("assets/icons/swap.svg", width: 32, height: 32),
                // ),
                child: Stack(
                  children: [
                    Positioned(
                      top: 1,
                      child: Container(
                        width: 60,
                        height: 60,
                        child: ElevatedButton(
                          onPressed: () => onClick(),
                          style: ButtonStyle(
                            padding: MaterialStateProperty.all(EdgeInsets.zero),
                            shape: MaterialStateProperty.all(CircleBorder()),
                            backgroundColor: MaterialStateProperty.all(Color(0xff101212)),
                          ),
                          child: SvgPicture.asset("assets/icons/${icon}.svg", width: 18, height: 18, fit: BoxFit.cover),
                        ),
                      ),
                    )
                  ],
                ),
              ),
              ...text != null ? [
                  SizedBox(height: 8),
                  Text(text!, style: Theme.of(context).textTheme.bodySmall),
              ] : [],
            ],
          );
    }
}

class CircleIcon extends StatelessWidget
{
    final String icon;

    const CircleIcon({super.key, required this.icon});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: 52,
          height: 52,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.02),
            borderRadius: BorderRadius.circular(52),
            border: Border(
              top: BorderSide(color: Colors.white.withOpacity(0.1), width: 1),
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.white.withOpacity(0.15),
                offset: Offset(0, 0.25), // TODO: ?
                blurRadius: 0,
                spreadRadius: 0,
              ),
            ],
          ),
          child: Stack(
            children: [
              Positioned(
                top: 0,
                child: Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(52),
                    color: Color(0xff101212),
                  ),
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      SvgPicture.asset("assets/icons/${icon}.svg", width: 22, height: 22, fit: BoxFit.cover)
                    ],
                  ),
                ),
              )
            ],
          ),
        );
    }
}

class CircleIconButtonGroup extends StatelessWidget
{
    final List<CircleIconButton> children;

    CircleIconButtonGroup({Key? key, required this.children});

  @override
    Widget build(BuildContext context)
    {
        return Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(children.length, (index)
            {
                return Row(
                  children: [
                    children[index],
                    if (index < children.length - 1)
                      SizedBox(width: 40),
                  ],
                );
            }),
          );
    }
}


class BlurredCirclePainter extends CustomPainter {
  final Color color;
  final double blurSigma;

  BlurredCirclePainter({required this.color, required this.blurSigma});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = color
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, blurSigma);

    canvas.drawCircle(
      Offset(size.width / 2, size.height / 2),
      size.width / 2,
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

// TODO: --

class RadialGradientDivider extends StatelessWidget {
  final double width;

  const RadialGradientDivider({Key? key, this.width = 350}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: 2,
      child: CustomPaint(
        painter: RadialGradientDividerPainter(),
      ),
    );
  }
}

class RadialGradientDividerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..shader = RadialGradient(
        center: Alignment.center,
        radius: 0.5,
        colors: [
          Colors.white.withOpacity(0.6),
          Colors.white.withOpacity(0),
        ],
        stops: [0, 1],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height))
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;

    canvas.drawLine(
      Offset(0, size.height / 2),
      Offset(size.width, size.height / 2),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}


class CustomDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(double.infinity, 2),
      painter: _DividerPainter(),
    );
  }
}

class _DividerPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..shader = RadialGradient(
        center: Alignment(0, 0),
        radius: 1,
        colors: [
          Colors.white.withOpacity(0.6),
          Colors.white.withOpacity(0),
        ],
        stops: [0, 1],
        transform: GradientRotation(1.5),
      ).createShader(Rect.fromCircle(center: Offset(size.width / 2, 1.5), radius: 0.5))
      ..strokeWidth = 2;

    canvas.drawLine(Offset(0, 1), Offset(size.width, 1), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}

class PyrinListViewItemBuilder<T>
{
    final String title;
    final String subtitle;
    final Widget? leading;
    final Widget? icon;

    PyrinListViewItemBuilder({
      required this.title,
      required this.subtitle,
      this.leading,
      this.icon,
});
}

class PyrinListViewItem extends StatelessWidget
{
    final String title;
    final String subtitle;
    final Widget? leading;
    final Widget? icon;
    final bool padding;

    PyrinListViewItem({
      required this.title,
      required this.subtitle,
      this.leading,
      this.icon,
      this.padding = true,
    });

    @override
    Widget build(BuildContext context)
    {
        return Container(
          padding: padding ? EdgeInsets.symmetric(vertical: 12) : null,
          color: Colors.transparent,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              ...icon != null ? [
                icon!,
                const SizedBox(width: 12),
              ] : [],
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(title, style: Theme.of(context).textTheme.bodyMedium),
                  Text(subtitle, style: Theme.of(context).textTheme.bodySmall!.copyWith(fontSize: 11, color: PyrinColors.LIGHT_TEXT_COLOR)),
                ],
              ),
              Flexible(child: Container()),
              ...leading != null ? [
                DefaultTextStyle.merge(
                  style: Theme.of(context).textTheme.bodyLarge!.copyWith(fontSize: 16, fontWeight: FontWeight.w600),
                  child: leading!,
                ),
              ] : []
            ],
          ),
        );

    }
}

class PyrinDivider extends StatelessWidget
{
    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: MediaQuery.sizeOf(context).width,
          height: 2,
          child: SvgPicture.asset("assets/home_list_divider.svg", fit: BoxFit.cover),
        );
    }
}

class PyrinListView<T> extends StatelessWidget
{
    final List<T> items;
    final PyrinListViewItemBuilder Function(BuildContext context, T item) itemBuilder;
    final void Function(T item)? onItemClick;

    PyrinListView({required this.items, required this.itemBuilder, this.onItemClick});

    @override
    Widget build(BuildContext context)
    {
        return ListView.separated(
          itemCount: items.length,
          shrinkWrap: true,
          separatorBuilder: (context, index) => PyrinDivider(),
          itemBuilder: (BuildContext context, int index)
          {
              final item = itemBuilder(context, items[index]);

              return GestureDetector(
                onTap: ()
                {
                    if (onItemClick != null)
                        onItemClick!(items[index]);
                },
                child: PyrinListViewItem(
                  title: item.title,
                  subtitle: item.subtitle,
                  leading: item.leading,
                  icon: item.icon,
                ),
              );
          },
        );
    }
}

void pyrinShowModalBottomSheet({
  required BuildContext context,
  required Widget child,
  bool isScrollControlled = false,
  bool isDismissible = false,
  double maxHeightRatio = 0.9,
})
{
    showModalBottomSheet(
      context: context,
      isScrollControlled: isScrollControlled,
      isDismissible: isDismissible,
      backgroundColor: Colors.transparent,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(30),
        ),
      ),
      builder: (BuildContext context)
      {
        return ConstrainedBox(
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * maxHeightRatio,
          ),
          child: Container(
            decoration: const BoxDecoration(
              color: PyrinColors.BACKGROUND_COLOR,
              borderRadius: const BorderRadius.vertical(
                top: Radius.circular(30),
              ),
            ),
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Container(
                  width: 90,
                  height: 4,
                  decoration: BoxDecoration(
                    color: PyrinColors.WHITE_COLOR,
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
                const SizedBox(height: 20),
                child
              ],
            ),
          ),
        );
      },
    );
}

class Avatar extends StatelessWidget
{
    final double size;

    Avatar({this.size = 40});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: size,
          height: size,
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
            borderRadius: BorderRadius.circular(size),
          ),
          child: Center(
            child: SvgPicture.asset(
              "assets/icons/avatar.svg",
              width: size / 2,
              height: size / 2,
              color: PyrinColors.LIGHT_TEXT_COLOR,
            ),
          ),
        );
    }
}

class CustomModal extends StatelessWidget
{
    final Widget child;
    final String? title;

    CustomModal({required this.child, this.title});

    @override
    Widget build(BuildContext context)
    {
        return Container(
          width: double.infinity,
          height: double.infinity,
          color: Colors.transparent,
          padding: EdgeInsets.all(20),
          child: Stack(
            alignment: Alignment.center,
            children: [
              PyrinCard(
                borderRadius: 30,
                child: Container(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Align(
                      //   alignment: Alignment.centerRight,
                      //   child: IconButton(
                      //     icon: SvgPicture.asset("assets/icons/close.svg"),
                      //     onPressed: () => Navigator.of(context).pop(),
                      //   ),
                      // ),
                      ...title != null ? [
                        Container(
                          margin: EdgeInsets.only(bottom: 20),
                          child: Text(title!, style: Theme.of(context).textTheme.bodyLarge),
                        ),
                      ] : [],
                      child,
                    ],
                  ),
                ),
              )
            ],
          ),
        );
    }
}

class Address extends StatelessWidget
{
    final String address;

    Address({required this.address});

    @override
    Widget build(BuildContext context)
    {
        return Row(
          children: [
              Text(
                AddressBook.shortenAddress(address),
                style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                  fontWeight: FontWeight.w400,
                  fontSize: 11,
                  color: PyrinColors.TEXT_COLOR
                ),
              ),
              IconButton(
                padding: EdgeInsets.zero,
                icon: SvgPicture.asset("assets/icons/copy.svg", width: 16, height: 16, colorFilter: ColorFilter.mode(PyrinColors.TEXT_COLOR, BlendMode.srcIn)),
                onPressed: () => Clipboard.setData(ClipboardData(text: address)),
              ),
          ],
        );
    }
}

class PriceBadge extends StatelessWidget
{
    final double value;

    PriceBadge({required this.value});

    @override
    Widget build(BuildContext context)
    {
        Color color = value > 0 ? PyrinColors.GREEN_COLOR : PyrinColors.RED_COLOR;

        return Container(
          padding: EdgeInsets.symmetric(vertical: 4, horizontal: 10),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(18),
          ),
          child: Text(
            "${value.toStringAsFixed(2)}%",
            style: Theme.of(context).textTheme.bodyMedium!.copyWith(
              fontWeight: FontWeight.w400,
              fontSize: 14,
              color: color,
              letterSpacing: 0.035
            ),
          ),
        );
    }
}

class PyrinPrimaryCircleButton extends StatelessWidget
{
    final double size;
    final String icon;
    final double iconSize;
    final VoidCallback onClick;

    PyrinPrimaryCircleButton({
      Key? key,
      this.size = 70,
      required this.icon,
      this.iconSize = 24,
      required this.onClick,
});

    @override
    Widget build(BuildContext context)
    {
        return ElevatedButton(
          style: ElevatedButton.styleFrom(
            // padding: EdgeInsets.all(27.5),
            foregroundColor: PyrinColors.BACKGROUND_COLOR,
            backgroundColor: PyrinColors.TEXT_COLOR,
            shape: CircleBorder(),
            elevation: 0,
            shadowColor: Colors.transparent,
          ),
          onPressed: onClick,
          child: Container(
              width: size,
              height: size,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SvgPicture.asset(
                    "assets/icons/$icon.svg",
                    width: iconSize,
                    height: iconSize,
                    fit: BoxFit.cover,
                    colorFilter: ColorFilter.mode(
                      PyrinColors.BACKGROUND_COLOR,
                      BlendMode.srcIn,
                    ),
                  )
                ],
              )
          ),
        );
    }
}

void showModal({
  required BuildContext context,
  required Widget child,
  String? title,
})
{
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      enableDrag: false,
      backgroundColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(20.0)),
      ),
      builder: (BuildContext context)
      {
          return CustomModal(
            child: child,
            title: title,
          );
      },
    );
}

void scanAddress(BuildContext context, Function(String address) callback)
{
  final size = MediaQuery.of(context).size.width - 20 * 2;
  final height = MediaQuery.of(context).size.height;
  final containerHeight = size * 1.5;

  pyrinShowModalBottomSheet(
      context: context,
      isScrollControlled: true,
      maxHeightRatio: (containerHeight + 70) / height,
      child: Container(
        width: double.infinity,
        height: containerHeight,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Scan QR Code", style: Theme.of(context).textTheme.bodyMedium!.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(height: 25),
            // TODO: Design / Dashed
            Container(
              width: size,
              height: size,
              child: MobileScanner(
                onDetect: (BarcodeCapture barcodes)
                {
                    final _barcode = barcodes.barcodes.firstOrNull;

                    if (_barcode?.rawValue != null)
                    {
                        // Close the modal
                        Navigator.pop(context);

                        callback(_barcode!.rawValue!);
                    }
                },
              ),
            ),
            // Container(
            //   width: double.infinity,
            //   height: 200,
            //   decoration: BoxDecoration(
            //     color: Colors.white.withOpacity(0.1),
            //     border: Border.all(color: Colors.white.withOpacity(0.1)),
            //   ),
            // ),
            Expanded(child: const SizedBox(height: 25)),
            PyrinElevatedButton(
                text: "Close",
                onClick: () => Navigator.pop(context),
                wide: true
            ),
          ],
        ),
      )

  );
}

void returnHome(BuildContext context)
{
    Navigator.of(context).popUntil((route) => route.isFirst);
}

// TODO: transaction id
void transactionConfirmedModal(BuildContext context)
{
  showModal(
    context: context,
    title: "Transaction Confirmed",
    child: Column(
      children: [
        // TODO: close-circle.svg when failed
        SvgPicture.asset("assets/tick-circle.svg", width: 100, height: 100),
        const SizedBox(height: 20),
        Text("Transaction is submitted successfully", style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: 30),
        PyrinDivider(),
        const SizedBox(height: 40),
        Container(
          width: MediaQuery.sizeOf(context).width * 0.35,
          child: PyrinElevatedButton(
            wide: true,
            text: "Done",
            onClick: ()
            {
                returnHome(context);
            },
          ),
        ),
      ],
    ),
  );
}
